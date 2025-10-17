import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Function

# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dataset class for PACS
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

                if train:
                    selected_images = images[:split_idx]
                else:
                    selected_images = images[split_idx:]

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

# Gradient Reversal Layer for DANN
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

# MMD loss for DAN with RBF kernel
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
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:source_batch_size, :source_batch_size]
    YY = kernels[source_batch_size:, source_batch_size:]
    XY = kernels[:source_batch_size, source_batch_size:]
    YX = kernels[source_batch_size:, :source_batch_size]

    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss

# ============================================================
# TASK 2: Domain-Alignment Based Adaptation Methods
# ============================================================

# Method 1: Deep Adaptation Network (DAN) - Statistical Alignment using MMD
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

# Method 2: Domain-Adversarial Neural Network (DANN) - Adversarial Alignment
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

# Method 3: Conditional Domain Adversarial Network (CDAN) - Class-Aware Alignment
class CDANModel(nn.Module):
    def __init__(self, num_classes=7):
        super(CDANModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(in_features, num_classes)

        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features * num_classes, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x, lambda_=1.0):
        features = self.backbone(x)
        class_output = self.classifier(features)

        softmax_output = nn.Softmax(dim=1)(class_output)
        feature_class = torch.bmm(softmax_output.unsqueeze(2), features.unsqueeze(1))
        feature_class = feature_class.view(feature_class.size(0), -1)

        reversed_features = self.grl(feature_class, lambda_)
        domain_output = self.domain_classifier(reversed_features)

        return class_output, domain_output

def train_dan(model, source_loader, target_loader, optimizer, scheduler, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_cls_loss = 0.0
        running_mmd_loss = 0.0
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
            running_cls_loss += cls_loss.item()
            running_mmd_loss += mmd.item()

        scheduler.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(source_loader):.4f}, Cls: {running_cls_loss/len(source_loader):.4f}, MMD: {running_mmd_loss/len(source_loader):.4f}')

def train_dann(model, source_loader, target_loader, optimizer, scheduler, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_cls_loss = 0.0
        running_domain_loss = 0.0
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
            running_cls_loss += cls_loss.item()
            running_domain_loss += domain_loss.item()

        scheduler.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(source_loader):.4f}, Cls: {running_cls_loss/len(source_loader):.4f}, Domain: {running_domain_loss/len(source_loader):.4f}')

def train_cdan(model, source_loader, target_loader, optimizer, scheduler, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_cls_loss = 0.0
        running_domain_loss = 0.0
        running_ent_loss = 0.0
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

            source_class_output, source_domain_output = model(source_images, lambda_)
            target_class_output, target_domain_output = model(target_images, lambda_)

            cls_loss = criterion(source_class_output, source_labels)

            source_domain_labels = torch.zeros(source_images.size(0), dtype=torch.long).to(device)
            target_domain_labels = torch.ones(target_images.size(0), dtype=torch.long).to(device)

            domain_loss = domain_criterion(source_domain_output, source_domain_labels) + \
                         domain_criterion(target_domain_output, target_domain_labels)

            p_t = nn.Softmax(dim=1)(target_class_output)
            ent_loss = -torch.mean(torch.sum(p_t * torch.log(p_t + 1e-8), 1))

            total_loss = cls_loss + lambda_ * domain_loss + 0.01 * ent_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_domain_loss += domain_loss.item()
            running_ent_loss += ent_loss.item()

        scheduler.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(source_loader):.4f}, Cls: {running_cls_loss/len(source_loader):.4f}, Domain: {running_domain_loss/len(source_loader):.4f}, Ent: {running_ent_loss/len(source_loader):.4f}')

def evaluate_dan(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs, _ = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels

def evaluate_dann(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs, _, _ = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels

def evaluate_cdan(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs, _ = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels

def calculate_rare_class_f1(all_labels, all_preds, target_loader):
    labels_array = np.array(target_loader.dataset.labels)
    class_counts = np.bincount(labels_array)
    rare_classes = np.argsort(class_counts)[:3]

    rare_true = []
    rare_pred = []
    for i, label in enumerate(all_labels):
        if label in rare_classes:
            rare_true.append(label)
            rare_pred.append(all_preds[i])

    if len(rare_true) > 0:
        f1 = f1_score(rare_true, rare_pred, average='macro', labels=rare_classes)
    else:
        f1 = 0.0

    return f1, rare_classes

def compute_per_class_accuracy(all_labels, all_preds, num_classes=7):
    """Compute accuracy for each class"""
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for label, pred in zip(all_labels, all_preds):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1

    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[i] = class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0

    return class_accuracies

def extract_features(model, loader, device, model_type='dan'):
    """Extract features from a model for proxy A-distance computation"""
    model.eval()
    all_features = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            if model_type == 'dan':
                _, features = model(images)
            elif model_type in ['dann', 'cdan']:
                if model_type == 'dann':
                    _, _, features = model(images)
                else:
                    _, features = model(images)  # For CDAN, we'll use class output as features

            all_features.append(features.cpu().numpy())

    return np.vstack(all_features)

def compute_proxy_a_distance(source_features, target_features):
    """
    Compute proxy A-distance using a binary classifier to distinguish
    between source and target features. Lower error = better alignment.
    Proxy A-distance = 2(1 - 2*error)
    """
    # Prepare data
    X = np.vstack([source_features, target_features])
    y = np.hstack([np.zeros(len(source_features)), np.ones(len(target_features))])

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Split into train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train a linear classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Compute error
    error = 1 - clf.score(X_test, y_test)

    # Compute proxy A-distance
    proxy_a_dist = 2 * (1 - 2 * error)

    return proxy_a_dist, error

def extract_features_with_labels(model, loader, device, model_type='dan', max_samples=2000):
    """Extract features for t-SNE visualization"""
    model.eval()
    features_list = []
    labels_list = []
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            if count >= max_samples:
                break
            images = images.to(device)
            if model_type == 'dan':
                _, features = model(images)
            elif model_type == 'dann':
                _, _, features = model(images)
            elif model_type == 'cdan':
                outputs, _ = model(images)
                features = outputs  # Use class output as features for CDAN

            features = features.view(features.size(0), -1)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            count += len(labels)

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    return features[:max_samples], labels[:max_samples]

def plot_tsne(source_features, source_labels, target_features, target_labels,
              title, filename, class_names):
    """Plot t-SNE visualization"""
    print(f'Computing t-SNE for {title}...')

    all_features = np.vstack([source_features, target_features])
    all_labels = np.concatenate([source_labels, target_labels])
    domains = np.array(['Source'] * len(source_features) + ['Target'] * len(target_features))

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(all_features)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Color by domain
    for domain in ['Source', 'Target']:
        mask = domains == domain
        axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1],
                       label=domain, alpha=0.6, s=20)
    axes[0].set_title(f'{title} - Colored by Domain')
    axes[0].legend()
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    # Plot 2: Color by class
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, class_name in enumerate(class_names):
        mask = all_labels == i
        axes[1].scatter(features_2d[mask, 0], features_2d[mask, 1],
                       label=class_name, alpha=0.6, s=20, color=colors[i])
    axes[1].set_title(f'{title} - Colored by Class')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.tight_layout()
    os.makedirs('Figures', exist_ok=True)
    plt.savefig(f'Figures/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: Figures/{filename}')

def plot_class_distribution_shift(source_loader, target_loader,
                                   source_class_acc, target_class_acc,
                                   class_names, filename):
    """Plot class distribution shift and accuracy comparison"""
    source_labels = np.array(source_loader.dataset.labels)
    target_labels = np.array(target_loader.dataset.labels)

    source_counts = np.bincount(source_labels, minlength=len(class_names))
    target_counts = np.bincount(target_labels, minlength=len(class_names))

    source_freq = source_counts / source_counts.sum()
    target_freq = target_counts / target_counts.sum()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Class frequency
    x = np.arange(len(class_names))
    width = 0.35
    axes[0].bar(x - width/2, source_freq, width, label='Source', alpha=0.8)
    axes[0].bar(x + width/2, target_freq, width, label='Target', alpha=0.8)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Class Distribution: Source vs Target')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Accuracy comparison
    source_acc_list = [source_class_acc[i] for i in range(len(class_names))]
    target_acc_list = [target_class_acc[i] for i in range(len(class_names))]

    axes[1].plot(x, source_acc_list, 'o-', label='Source Accuracy', linewidth=2, markersize=8)
    axes[1].plot(x, target_acc_list, 's-', label='Target Accuracy', linewidth=2, markersize=8)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Per-Class Accuracy: Source vs Target')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim([0, 1.05])
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs('Figures', exist_ok=True)
    plt.savefig(f'Figures/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: Figures/{filename}')

def plot_confusion_matrix(all_labels, all_preds, method_name, domain_name, class_names=None):
    """Plot and save confusion matrix"""
    if class_names is None:
        class_names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{method_name} - {domain_name} Domain Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs('Figures', exist_ok=True)
    plt.savefig(f'Figures/Task2-{method_name}-{domain_name}-confusion-matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: Figures/Task2-{method_name}-{domain_name}-confusion-matrix.png')

    return cm

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

    data_root = './data/PACS'
    source_domain = 'photo'
    target_domain = 'art_painting'
    batch_size = 32
    num_classes = 7
    epochs = 50

    print(f'\nSource Domain: {source_domain}')
    print(f'Target Domain: {target_domain}\n')

    source_train_dataset = PACSDataset(data_root, source_domain, transform=train_transform, train=True)
    target_train_dataset = PACSDataset(data_root, target_domain, transform=train_transform, train=True)
    target_test_dataset = PACSDataset(data_root, target_domain, transform=test_transform, train=False)

    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create source test loader for source domain evaluation
    source_test_dataset = PACSDataset(data_root, source_domain, transform=test_transform, train=False)
    source_test_loader = DataLoader(source_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    results = {}
    class_names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    # Method 1: DAN
    print('='*60)
    print('Training DAN (Deep Adaptation Network)...')
    print('='*60)
    dan_model = DANModel(num_classes=num_classes).to(device)
    dan_optimizer = optim.SGD(dan_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    dan_scheduler = optim.lr_scheduler.StepLR(dan_optimizer, step_size=20, gamma=0.1)
    train_dan(dan_model, source_train_loader, target_train_loader, dan_optimizer, dan_scheduler, device, epochs=epochs)

    # Evaluate on target domain
    dan_target_acc, dan_target_preds, dan_target_labels = evaluate_dan(dan_model, target_test_loader, device)
    dan_f1, rare_classes = calculate_rare_class_f1(dan_target_labels, dan_target_preds, target_test_loader)

    # Evaluate on source domain
    dan_source_acc, dan_source_preds, dan_source_labels = evaluate_dan(dan_model, source_test_loader, device)

    # Compute per-class accuracy for both domains
    dan_target_class_acc = compute_per_class_accuracy(dan_target_labels, dan_target_preds, num_classes)
    dan_source_class_acc = compute_per_class_accuracy(dan_source_labels, dan_source_preds, num_classes)

    # Extract features and compute proxy A-distance
    print('Computing proxy A-distance for DAN...')
    dan_source_features = extract_features(dan_model, source_test_loader, device, 'dan')
    dan_target_features = extract_features(dan_model, target_test_loader, device, 'dan')
    dan_proxy_a, dan_error = compute_proxy_a_distance(dan_source_features, dan_target_features)

    # Generate visualizations
    print('\nGenerating visualizations for DAN...')
    dan_source_feat, dan_source_feat_labels = extract_features_with_labels(dan_model, source_test_loader, device, 'dan')
    dan_target_feat, dan_target_feat_labels = extract_features_with_labels(dan_model, target_test_loader, device, 'dan')
    plot_tsne(dan_source_feat, dan_source_feat_labels, dan_target_feat, dan_target_feat_labels,
              'Task 2: DAN', 'Task2-DAN-tsne.png', class_names)
    plot_confusion_matrix(dan_target_labels, dan_target_preds, 'DAN', 'Target', class_names)
    plot_confusion_matrix(dan_source_labels, dan_source_preds, 'DAN', 'Source', class_names)
    plot_class_distribution_shift(source_test_loader, target_test_loader,
                                   dan_source_class_acc, dan_target_class_acc, class_names,
                                   'Task2-DAN-class-distribution.png')

    results['DAN'] = {
        'target_accuracy': dan_target_acc,
        'source_accuracy': dan_source_acc,
        'rare_f1': dan_f1,
        'target_class_acc': dan_target_class_acc,
        'source_class_acc': dan_source_class_acc,
        'proxy_a_distance': dan_proxy_a,
        'domain_classifier_error': dan_error
    }

    print(f'\n--- DAN Results ---')
    print(f'Target Domain Accuracy: {dan_target_acc*100:.2f}%')
    print(f'Source Domain Accuracy: {dan_source_acc*100:.2f}%')
    print(f'Rare Classes F1-Score: {dan_f1:.4f}')
    print(f'Proxy A-Distance: {dan_proxy_a:.4f} (Domain Classifier Error: {dan_error:.4f})')
    print(f'Per-Class Accuracy (Target): {[f"{acc*100:.1f}%" for acc in dan_target_class_acc.values()]}')

    # Method 2: DANN
    print('\n' + '='*60)
    print('Training DANN (Domain-Adversarial Neural Network)...')
    print('='*60)
    dann_model = DANNModel(num_classes=num_classes).to(device)
    dann_optimizer = optim.SGD(dann_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    dann_scheduler = optim.lr_scheduler.StepLR(dann_optimizer, step_size=20, gamma=0.1)
    train_dann(dann_model, source_train_loader, target_train_loader, dann_optimizer, dann_scheduler, device, epochs=epochs)

    # Evaluate on target domain
    dann_target_acc, dann_target_preds, dann_target_labels = evaluate_dann(dann_model, target_test_loader, device)
    dann_f1, _ = calculate_rare_class_f1(dann_target_labels, dann_target_preds, target_test_loader)

    # Evaluate on source domain
    dann_source_acc, dann_source_preds, dann_source_labels = evaluate_dann(dann_model, source_test_loader, device)

    # Compute per-class accuracy for both domains
    dann_target_class_acc = compute_per_class_accuracy(dann_target_labels, dann_target_preds, num_classes)
    dann_source_class_acc = compute_per_class_accuracy(dann_source_labels, dann_source_preds, num_classes)

    # Extract features and compute proxy A-distance
    print('Computing proxy A-distance for DANN...')
    dann_source_features = extract_features(dann_model, source_test_loader, device, 'dann')
    dann_target_features = extract_features(dann_model, target_test_loader, device, 'dann')
    dann_proxy_a, dann_error = compute_proxy_a_distance(dann_source_features, dann_target_features)

    # Generate visualizations
    print('\nGenerating visualizations for DANN...')
    dann_source_feat, dann_source_feat_labels = extract_features_with_labels(dann_model, source_test_loader, device, 'dann')
    dann_target_feat, dann_target_feat_labels = extract_features_with_labels(dann_model, target_test_loader, device, 'dann')
    plot_tsne(dann_source_feat, dann_source_feat_labels, dann_target_feat, dann_target_feat_labels,
              'Task 2: DANN', 'Task2-DANN-tsne.png', class_names)
    plot_confusion_matrix(dann_target_labels, dann_target_preds, 'DANN', 'Target', class_names)
    plot_confusion_matrix(dann_source_labels, dann_source_preds, 'DANN', 'Source', class_names)
    plot_class_distribution_shift(source_test_loader, target_test_loader,
                                   dann_source_class_acc, dann_target_class_acc, class_names,
                                   'Task2-DANN-class-distribution.png')

    results['DANN'] = {
        'target_accuracy': dann_target_acc,
        'source_accuracy': dann_source_acc,
        'rare_f1': dann_f1,
        'target_class_acc': dann_target_class_acc,
        'source_class_acc': dann_source_class_acc,
        'proxy_a_distance': dann_proxy_a,
        'domain_classifier_error': dann_error
    }

    print(f'\n--- DANN Results ---')
    print(f'Target Domain Accuracy: {dann_target_acc*100:.2f}%')
    print(f'Source Domain Accuracy: {dann_source_acc*100:.2f}%')
    print(f'Rare Classes F1-Score: {dann_f1:.4f}')
    print(f'Proxy A-Distance: {dann_proxy_a:.4f} (Domain Classifier Error: {dann_error:.4f})')
    print(f'Per-Class Accuracy (Target): {[f"{acc*100:.1f}%" for acc in dann_target_class_acc.values()]}')

    # Method 3: CDAN
    print('\n' + '='*60)
    print('Training CDAN (Conditional Domain Adversarial Network)...')
    print('='*60)
    cdan_model = CDANModel(num_classes=num_classes).to(device)
    cdan_optimizer = optim.SGD(cdan_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    cdan_scheduler = optim.lr_scheduler.StepLR(cdan_optimizer, step_size=20, gamma=0.1)
    train_cdan(cdan_model, source_train_loader, target_train_loader, cdan_optimizer, cdan_scheduler, device, epochs=epochs)

    # Evaluate on target domain
    cdan_target_acc, cdan_target_preds, cdan_target_labels = evaluate_cdan(cdan_model, target_test_loader, device)
    cdan_f1, _ = calculate_rare_class_f1(cdan_target_labels, cdan_target_preds, target_test_loader)

    # Evaluate on source domain
    cdan_source_acc, cdan_source_preds, cdan_source_labels = evaluate_cdan(cdan_model, source_test_loader, device)

    # Compute per-class accuracy for both domains
    cdan_target_class_acc = compute_per_class_accuracy(cdan_target_labels, cdan_target_preds, num_classes)
    cdan_source_class_acc = compute_per_class_accuracy(cdan_source_labels, cdan_source_preds, num_classes)

    # Extract features and compute proxy A-distance
    print('Computing proxy A-distance for CDAN...')
    cdan_source_features = extract_features(cdan_model, source_test_loader, device, 'cdan')
    cdan_target_features = extract_features(cdan_model, target_test_loader, device, 'cdan')
    cdan_proxy_a, cdan_error = compute_proxy_a_distance(cdan_source_features, cdan_target_features)

    # Generate visualizations
    print('\nGenerating visualizations for CDAN...')
    cdan_source_feat, cdan_source_feat_labels = extract_features_with_labels(cdan_model, source_test_loader, device, 'cdan')
    cdan_target_feat, cdan_target_feat_labels = extract_features_with_labels(cdan_model, target_test_loader, device, 'cdan')
    plot_tsne(cdan_source_feat, cdan_source_feat_labels, cdan_target_feat, cdan_target_feat_labels,
              'Task 2: CDAN', 'Task2-CDAN-tsne.png', class_names)
    plot_confusion_matrix(cdan_target_labels, cdan_target_preds, 'CDAN', 'Target', class_names)
    plot_confusion_matrix(cdan_source_labels, cdan_source_preds, 'CDAN', 'Source', class_names)
    plot_class_distribution_shift(source_test_loader, target_test_loader,
                                   cdan_source_class_acc, cdan_target_class_acc, class_names,
                                   'Task2-CDAN-class-distribution.png')

    results['CDAN'] = {
        'target_accuracy': cdan_target_acc,
        'source_accuracy': cdan_source_acc,
        'rare_f1': cdan_f1,
        'target_class_acc': cdan_target_class_acc,
        'source_class_acc': cdan_source_class_acc,
        'proxy_a_distance': cdan_proxy_a,
        'domain_classifier_error': cdan_error
    }

    print(f'\n--- CDAN Results ---')
    print(f'Target Domain Accuracy: {cdan_target_acc*100:.2f}%')
    print(f'Source Domain Accuracy: {cdan_source_acc*100:.2f}%')
    print(f'Rare Classes F1-Score: {cdan_f1:.4f}')
    print(f'Proxy A-Distance: {cdan_proxy_a:.4f} (Domain Classifier Error: {cdan_error:.4f})')
    print(f'Per-Class Accuracy (Target): {[f"{acc*100:.1f}%" for acc in cdan_target_class_acc.values()]}')

    # Summary
    print('\n' + '='*80)
    print('COMPREHENSIVE SUMMARY: Domain Adaptation Methods Comparison')
    print('='*80)
    print(f'\nRare Classes (3 rarest): {[class_names[i] for i in rare_classes]}')
    print(f'Class Names: {class_names}\n')

    for method, metrics in results.items():
        print(f'\n{"-"*80}')
        print(f'{method} Results:')
        print(f'{"-"*80}')
        print(f'  Target Domain Accuracy: {metrics["target_accuracy"]*100:.2f}%')
        print(f'  Source Domain Accuracy: {metrics["source_accuracy"]*100:.2f}%')
        print(f'  Accuracy Drop (Source→Target): {(metrics["source_accuracy"]-metrics["target_accuracy"])*100:+.2f}%')
        print(f'  Rare Classes F1-Score: {metrics["rare_f1"]:.4f}')
        print(f'  Proxy A-Distance: {metrics["proxy_a_distance"]:.4f}')
        print(f'  Domain Classifier Error: {metrics["domain_classifier_error"]:.4f}')
        print(f'\n  Per-Class Accuracy (Target):')
        for i, (class_name, acc) in enumerate(zip(class_names, metrics["target_class_acc"].values())):
            print(f'    {class_name:12s}: {acc*100:5.1f}%')
        print(f'\n  Per-Class Accuracy (Source):')
        for i, (class_name, acc) in enumerate(zip(class_names, metrics["source_class_acc"].values())):
            print(f'    {class_name:12s}: {acc*100:5.1f}%')

    # Alignment-Discrimination Trade-off Analysis
    print(f'\n{"="*80}')
    print('ALIGNMENT-DISCRIMINATION TRADE-OFF ANALYSIS')
    print('='*80)
    print('\nInterpretation:')
    print('- Lower Proxy A-Distance = Better domain alignment')
    print('- Lower Domain Classifier Error = Worse domain alignment (domains are distinguishable)')
    print('- Source accuracy drop may indicate loss of discriminative features due to alignment\n')

    for method, metrics in results.items():
        source_drop = (metrics["source_accuracy"] - metrics["target_accuracy"]) * 100
        print(f'{method}:')
        print(f'  Domain Alignment (Proxy A-dist): {metrics["proxy_a_distance"]:.4f}')
        print(f'  Alignment Quality: {"Good" if metrics["domain_classifier_error"] > 0.4 else "Moderate" if metrics["domain_classifier_error"] > 0.3 else "Weak"}')
        if source_drop > 5:
            print(f'  ⚠️  Source accuracy maintained ({metrics["source_accuracy"]*100:.1f}%), no discriminative feature loss')
        elif source_drop > -5:
            print(f'  ✓  Balanced trade-off (source drop: {source_drop:.1f}%)')
        else:
            print(f'  ⚠️  Significant source accuracy drop ({source_drop:.1f}%), possible over-alignment')
        print()

    torch.save(dan_model.state_dict(), 'dan_model.pth')
    torch.save(dann_model.state_dict(), 'dann_model.pth')
    torch.save(cdan_model.state_dict(), 'cdan_model.pth')
    print('\nModels saved.')
    print('All visualizations saved in Figures/ directory')

if __name__ == '__main__':
    main()
