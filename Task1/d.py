import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Function

# Fix random seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# DATASET CLASS
# ============================================================

class PACSDataset(Dataset):
    def __init__(self, root_dir, domain, transform=None, train=True):
        self.root_dir = root_dir
        self.domain = domain
        self.transform = transform
        self.samples = []
        self.labels = []

        domain_path = os.path.join(root_dir, domain)
        classes = sorted(os.listdir(domain_path))

        for idx, class_name in enumerate(classes):
            class_path = os.path.join(domain_path, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                split_idx = int(len(images) * 0.8)
                selected_images = images[:split_idx] if train else images[split_idx:]

                for img_name in selected_images:
                    self.samples.append(os.path.join(class_path, img_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ============================================================
# MODEL DEFINITIONS
# ============================================================

class SourceModel(nn.Module):
    def __init__(self, num_classes=7):
        super(SourceModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class DANModel(nn.Module):
    def __init__(self, num_classes=7):
        super(DANModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.bottleneck = nn.Linear(in_features, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        bottleneck_features = self.bottleneck(features)
        outputs = self.classifier(bottleneck_features)
        return outputs, bottleneck_features

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)

class DANNModel(nn.Module):
    def __init__(self, num_classes=7):
        super(DANNModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x, lambda_=1.0):
        features = self.backbone(x)
        class_output = self.classifier(features)
        reversed_features = self.grl(features, lambda_)
        domain_output = self.domain_classifier(reversed_features)
        return class_output, domain_output, features

# ============================================================
# MMD LOSS
# ============================================================

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source_batch_size = int(source.size(0))
    target_batch_size = int(target.size(0))
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    XX = kernels[:source_batch_size, :source_batch_size]
    YY = kernels[source_batch_size:, source_batch_size:]
    XY = kernels[:source_batch_size, source_batch_size:]
    YX = kernels[source_batch_size:, :source_batch_size]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss

# ============================================================
# LABEL SHIFT SIMULATION
# ============================================================

def create_label_shift_dataset(dataset, shift_type='remove_class', target_class=0, keep_ratio=0.1):
    """Create label shift by manipulating class distribution"""
    labels_array = np.array(dataset.labels)
    indices = np.arange(len(dataset))

    if shift_type == 'remove_class':
        # Rare class scenario: keep only keep_ratio of target_class
        class_indices = indices[labels_array == target_class]
        other_indices = indices[labels_array != target_class]
        keep_count = int(len(class_indices) * keep_ratio)
        np.random.shuffle(class_indices)
        kept_class_indices = class_indices[:keep_count]
        final_indices = np.concatenate([kept_class_indices, other_indices])
        print(f'  Rare class {target_class}: {len(class_indices)} -> {keep_count} samples ({keep_ratio*100:.0f}%)')

    elif shift_type == 'downsample_multiple':
        # Severe label shift: downsample multiple classes
        downsample_classes = [0, 1, 2]
        final_indices = []
        for cls in range(7):
            class_indices = indices[labels_array == cls]
            if cls in downsample_classes:
                keep_count = int(len(class_indices) * 0.3)
                np.random.shuffle(class_indices)
                final_indices.extend(class_indices[:keep_count])
                print(f'  Class {cls}: {len(class_indices)} -> {keep_count} samples (30%)')
            else:
                final_indices.extend(class_indices)
                print(f'  Class {cls}: {len(class_indices)} samples (100%)')
        final_indices = np.array(final_indices)
    else:
        final_indices = indices

    return Subset(dataset, final_indices.tolist())

def print_class_distribution(dataset, title, class_names):
    """Print class distribution"""
    if isinstance(dataset, Subset):
        labels = [dataset.dataset.labels[i] for i in dataset.indices]
    else:
        labels = dataset.labels
    labels_array = np.array(labels)
    print(f'\n{title}:')
    for i, class_name in enumerate(class_names):
        count = np.sum(labels_array == i)
        pct = 100 * count / len(labels_array) if len(labels_array) > 0 else 0
        print(f'  {class_name:12s}: {count:4d} samples ({pct:5.1f}%)')

# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_source_only(model, train_loader, optimizer, scheduler, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

def train_dan(model, source_loader, target_loader, optimizer, scheduler, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        target_iter = iter(target_loader)
        for source_images, source_labels in source_loader:
            try:
                target_images, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_images, _ = next(target_iter)
            source_images, source_labels = source_images.to(device), source_labels.to(device)
            target_images = target_images.to(device)
            optimizer.zero_grad()
            source_outputs, source_features = model(source_images)
            _, target_features = model(target_images)
            cls_loss = criterion(source_outputs, source_labels)
            mmd = mmd_loss(source_features, target_features)
            total_loss = cls_loss + 0.5 * mmd
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(source_loader):.4f}')

def train_dann(model, source_loader, target_loader, optimizer, scheduler, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        target_iter = iter(target_loader)
        p = float(epoch) / epochs
        lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
        for source_images, source_labels in source_loader:
            try:
                target_images, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_images, _ = next(target_iter)
            source_images, source_labels = source_images.to(device), source_labels.to(device)
            target_images = target_images.to(device)
            optimizer.zero_grad()
            source_class_output, source_domain_output, _ = model(source_images, lambda_)
            _, target_domain_output, _ = model(target_images, lambda_)
            cls_loss = criterion(source_class_output, source_labels)
            source_domain_labels = torch.zeros(source_images.size(0), dtype=torch.long).to(device)
            target_domain_labels = torch.ones(target_images.size(0), dtype=torch.long).to(device)
            domain_loss = domain_criterion(source_domain_output, source_domain_labels) + \
                         domain_criterion(target_domain_output, target_domain_labels)
            total_loss = cls_loss + lambda_ * domain_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(source_loader):.4f}')

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate(model, test_loader, device, model_type='source'):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            if model_type == 'source':
                outputs = model(images)
            elif model_type == 'dan':
                outputs, _ = model(images)
            elif model_type == 'dann':
                outputs, _, _ = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds), all_preds, all_labels

def compute_per_class_metrics(labels, preds, num_classes=7):
    class_acc, class_f1 = [], []
    for i in range(num_classes):
        mask = np.array(labels) == i
        if np.sum(mask) > 0:
            class_labels = np.array(labels)[mask]
            class_preds = np.array(preds)[mask]
            acc = accuracy_score(class_labels, class_preds)
            binary_labels = (class_labels == i).astype(int)
            binary_preds = (class_preds == i).astype(int)
            f1 = f1_score(binary_labels, binary_preds, zero_division=0) if np.sum(binary_labels) > 0 else 0.0
            class_acc.append(acc)
            class_f1.append(f1)
        else:
            class_acc.append(0.0)
            class_f1.append(0.0)
    return class_acc, class_f1

def compute_confusion_for_rare_class(labels, preds, rare_class):
    labels, preds = np.array(labels), np.array(preds)
    tp = np.sum((labels == rare_class) & (preds == rare_class))
    fp = np.sum((labels != rare_class) & (preds == rare_class))
    fn = np.sum((labels == rare_class) & (preds != rare_class))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_per_class_comparison(results, class_names, scenario_name):
    fig, ax = plt.subplots(figsize=(14, 6))
    x, width = np.arange(len(class_names)), 0.25
    for i, method in enumerate(results.keys()):
        ax.bar(x + i * width, results[method]['class_acc'], width, label=method, alpha=0.8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Per-Class Accuracy - {scenario_name}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    os.makedirs('Figures', exist_ok=True)
    plt.savefig(f'Figures/Task4-{scenario_name}-per-class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: Figures/Task4-{scenario_name}-per-class.png')

def plot_confusion_matrix(labels, preds, method_name, scenario_name, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{method_name} - {scenario_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs('Figures', exist_ok=True)
    plt.savefig(f'Figures/Task4-{scenario_name}-{method_name}-confusion.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_root, source_domain, target_domain = './data/PACS', 'photo', 'art_painting'
    batch_size, num_classes, epochs = 32, 7, 30
    class_names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    print(f'\nSource: {source_domain}, Target: {target_domain}\n')

    # Load datasets
    source_train_dataset = PACSDataset(data_root, source_domain, transform=train_transform, train=True)
    target_train_dataset = PACSDataset(data_root, target_domain, transform=train_transform, train=True)
    target_test_dataset = PACSDataset(data_root, target_domain, transform=test_transform, train=False)

    print('='*80)
    print('TASK 4: CONCEPT SHIFT AND RARE-CLASS SCENARIOS')
    print('='*80)

    # ============================================================
    # SCENARIO 1: BASELINE (No Shift)
    # ============================================================
    print('\n' + '='*80)
    print('SCENARIO 1: BASELINE (No Label Shift)')
    print('='*80)

    print_class_distribution(source_train_dataset, 'Source Training', class_names)
    print_class_distribution(target_test_dataset, 'Target Test', class_names)

    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    baseline_results = {}

    # Source-Only
    print('\n--- Source-Only (Baseline) ---')
    source_model = SourceModel(num_classes=num_classes).to(device)
    optimizer = optim.SGD(source_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_source_only(source_model, source_train_loader, optimizer, scheduler, device, epochs=epochs)
    acc, preds, labels = evaluate(source_model, target_test_loader, device, 'source')
    class_acc, class_f1 = compute_per_class_metrics(labels, preds, num_classes)
    baseline_results['Source-Only'] = {'acc': acc, 'class_acc': class_acc, 'preds': preds, 'labels': labels}
    print(f'Accuracy: {acc*100:.2f}%')

    # DAN
    print('\n--- DAN (Baseline) ---')
    dan_model = DANModel(num_classes=num_classes).to(device)
    optimizer = optim.SGD(dan_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_dan(dan_model, source_train_loader, target_train_loader, optimizer, scheduler, device, epochs=epochs)
    acc, preds, labels = evaluate(dan_model, target_test_loader, device, 'dan')
    class_acc, class_f1 = compute_per_class_metrics(labels, preds, num_classes)
    baseline_results['DAN'] = {'acc': acc, 'class_acc': class_acc, 'preds': preds, 'labels': labels}
    print(f'Accuracy: {acc*100:.2f}%')

    # DANN
    print('\n--- DANN (Baseline) ---')
    dann_model = DANNModel(num_classes=num_classes).to(device)
    optimizer = optim.SGD(dann_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_dann(dann_model, source_train_loader, target_train_loader, optimizer, scheduler, device, epochs=epochs)
    acc, preds, labels = evaluate(dann_model, target_test_loader, device, 'dann')
    class_acc, class_f1 = compute_per_class_metrics(labels, preds, num_classes)
    baseline_results['DANN'] = {'acc': acc, 'class_acc': class_acc, 'preds': preds, 'labels': labels}
    print(f'Accuracy: {acc*100:.2f}%')

    plot_per_class_comparison(baseline_results, class_names, 'Baseline')
    for method in baseline_results:
        plot_confusion_matrix(baseline_results[method]['labels'],
                            baseline_results[method]['preds'],
                            method, 'Baseline', class_names)

    # ============================================================
    # SCENARIO 2: RARE CLASS
    # ============================================================
    print('\n\n' + '='*80)
    print('SCENARIO 2: RARE CLASS (Class 0 "dog" -> 10%)')
    print('='*80)

    rare_class = 0
    target_rare_dataset = create_label_shift_dataset(target_train_dataset, 'remove_class',
                                                      target_class=rare_class, keep_ratio=0.1)
    print_class_distribution(target_rare_dataset, 'Target Training (Rare)', class_names)
    target_rare_loader = DataLoader(target_rare_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    rare_results = {}

    # Source-Only
    print('\n--- Source-Only (Rare) ---')
    source_model_rare = SourceModel(num_classes=num_classes).to(device)
    optimizer = optim.SGD(source_model_rare.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_source_only(source_model_rare, source_train_loader, optimizer, scheduler, device, epochs=epochs)
    acc, preds, labels = evaluate(source_model_rare, target_test_loader, device, 'source')
    class_acc, class_f1 = compute_per_class_metrics(labels, preds, num_classes)
    precision, recall = compute_confusion_for_rare_class(labels, preds, rare_class)
    rare_results['Source-Only'] = {'acc': acc, 'class_acc': class_acc, 'precision': precision,
                                    'recall': recall, 'preds': preds, 'labels': labels}
    print(f'Accuracy: {acc*100:.2f}%, Rare Class P: {precision:.4f}, R: {recall:.4f}')

    # DAN
    print('\n--- DAN (Rare) ---')
    dan_model_rare = DANModel(num_classes=num_classes).to(device)
    optimizer = optim.SGD(dan_model_rare.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_dan(dan_model_rare, source_train_loader, target_rare_loader, optimizer, scheduler, device, epochs=epochs)
    acc, preds, labels = evaluate(dan_model_rare, target_test_loader, device, 'dan')
    class_acc, class_f1 = compute_per_class_metrics(labels, preds, num_classes)
    precision, recall = compute_confusion_for_rare_class(labels, preds, rare_class)
    rare_results['DAN'] = {'acc': acc, 'class_acc': class_acc, 'precision': precision,
                            'recall': recall, 'preds': preds, 'labels': labels}
    print(f'Accuracy: {acc*100:.2f}%, Rare Class P: {precision:.4f}, R: {recall:.4f}')

    # DANN
    print('\n--- DANN (Rare) ---')
    dann_model_rare = DANNModel(num_classes=num_classes).to(device)
    optimizer = optim.SGD(dann_model_rare.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_dann(dann_model_rare, source_train_loader, target_rare_loader, optimizer, scheduler, device, epochs=epochs)
    acc, preds, labels = evaluate(dann_model_rare, target_test_loader, device, 'dann')
    class_acc, class_f1 = compute_per_class_metrics(labels, preds, num_classes)
    precision, recall = compute_confusion_for_rare_class(labels, preds, rare_class)
    rare_results['DANN'] = {'acc': acc, 'class_acc': class_acc, 'precision': precision,
                             'recall': recall, 'preds': preds, 'labels': labels}
    print(f'Accuracy: {acc*100:.2f}%, Rare Class P: {precision:.4f}, R: {recall:.4f}')

    plot_per_class_comparison(rare_results, class_names, 'RareClass')
    for method in rare_results:
        plot_confusion_matrix(rare_results[method]['labels'],
                            rare_results[method]['preds'],
                            method, 'RareClass', class_names)

    # ============================================================
    # SCENARIO 3: LABEL SHIFT
    # ============================================================
    print('\n\n' + '='*80)
    print('SCENARIO 3: LABEL SHIFT (Classes 0,1,2 -> 30%)')
    print('='*80)

    target_shift_dataset = create_label_shift_dataset(target_train_dataset, 'downsample_multiple')
    print_class_distribution(target_shift_dataset, 'Target Training (Shift)', class_names)
    target_shift_loader = DataLoader(target_shift_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    shift_results = {}

    # Source-Only
    print('\n--- Source-Only (Shift) ---')
    source_model_shift = SourceModel(num_classes=num_classes).to(device)
    optimizer = optim.SGD(source_model_shift.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_source_only(source_model_shift, source_train_loader, optimizer, scheduler, device, epochs=epochs)
    acc, preds, labels = evaluate(source_model_shift, target_test_loader, device, 'source')
    class_acc, class_f1 = compute_per_class_metrics(labels, preds, num_classes)
    shift_results['Source-Only'] = {'acc': acc, 'class_acc': class_acc, 'preds': preds, 'labels': labels}
    print(f'Accuracy: {acc*100:.2f}%')

    # DAN
    print('\n--- DAN (Shift) ---')
    dan_model_shift = DANModel(num_classes=num_classes).to(device)
    optimizer = optim.SGD(dan_model_shift.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_dan(dan_model_shift, source_train_loader, target_shift_loader, optimizer, scheduler, device, epochs=epochs)
    acc, preds, labels = evaluate(dan_model_shift, target_test_loader, device, 'dan')
    class_acc, class_f1 = compute_per_class_metrics(labels, preds, num_classes)
    shift_results['DAN'] = {'acc': acc, 'class_acc': class_acc, 'preds': preds, 'labels': labels}
    print(f'Accuracy: {acc*100:.2f}%')

    # DANN
    print('\n--- DANN (Shift) ---')
    dann_model_shift = DANNModel(num_classes=num_classes).to(device)
    optimizer = optim.SGD(dann_model_shift.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_dann(dann_model_shift, source_train_loader, target_shift_loader, optimizer, scheduler, device, epochs=epochs)
    acc, preds, labels = evaluate(dann_model_shift, target_test_loader, device, 'dann')
    class_acc, class_f1 = compute_per_class_metrics(labels, preds, num_classes)
    shift_results['DANN'] = {'acc': acc, 'class_acc': class_acc, 'preds': preds, 'labels': labels}
    print(f'Accuracy: {acc*100:.2f}%')

    plot_per_class_comparison(shift_results, class_names, 'LabelShift')
    for method in shift_results:
        plot_confusion_matrix(shift_results[method]['labels'],
                            shift_results[method]['preds'],
                            method, 'LabelShift', class_names)

    # ============================================================
    # SUMMARY
    # ============================================================
    print('\n\n' + '='*80)
    print('SUMMARY: CONCEPT SHIFT ANALYSIS')
    print('='*80)

    scenarios = {'Baseline': baseline_results, 'Rare Class': rare_results, 'Label Shift': shift_results}

    print(f'\n{"Scenario":<20} {"Source-Only":<15} {"DAN":<15} {"DANN":<15}')
    print('-'*80)
    for scenario_name, results in scenarios.items():
        s = results['Source-Only']['acc'] * 100
        d = results['DAN']['acc'] * 100
        a = results['DANN']['acc'] * 100
        print(f'{scenario_name:<20} {s:>6.2f}%         {d:>6.2f}%         {a:>6.2f}%')

    print(f'\n{"="*80}')
    print('NEGATIVE TRANSFER ANALYSIS')
    print('='*80)
    for scenario_name, results in scenarios.items():
        print(f'\n{scenario_name}:')
        source_only_acc = results['Source-Only']['acc']
        for method in ['DAN', 'DANN']:
            diff = (results[method]['acc'] - source_only_acc) * 100
            if diff < -2:
                status = '⚠️  NEGATIVE TRANSFER'
            elif diff > 2:
                status = '✓ Positive transfer'
            else:
                status = '≈ Neutral'
            print(f'  {method}: {diff:+.2f}% - {status}')

    print(f'\n{"="*80}')
    print('KEY FINDINGS')
    print('='*80)
    print('\n1. When Invariance Breaks Down:')
    print('   - DA methods assume Ps(Y) = Pt(Y) (same label distribution)')
    print('   - Rare class and label shift violate this assumption')
    print('   - Alignment can cause negative transfer\n')

    print('2. Negative Transfer Effects:')
    print('   - Aligning distributions when label distributions differ:')
    print('     * Confuses rare classes with common classes')
    print('     * Reduces precision/recall for underrepresented classes')
    print('     * Can cause overall accuracy degradation\n')

    print('3. Method Robustness:')
    print('   - Source-Only may outperform DA under severe label shift')
    print('   - DANN (adversarial) vs DAN (statistical) show different failure modes')
    print('   - Class-conditional methods may be more robust\n')

    print('All visualizations saved in Figures/')
    torch.save(source_model.state_dict(), 'task4_source_only_model.pth')
    torch.save(dan_model.state_dict(), 'task4_dan_model.pth')
    torch.save(dann_model.state_dict(), 'task4_dann_model.pth')

if __name__ == '__main__':
    main()
