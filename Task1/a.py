import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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

# Model with ResNet-50 backbone
class SourceModel(nn.Module):
    def __init__(self, num_classes=7):
        super(SourceModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """Extract features before final classification layer"""
        for name, module in self.backbone.named_children():
            if name == 'fc':
                break
            x = module(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
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

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels

def extract_features_with_labels(model, loader, device, max_samples=2000):
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
            features = model.extract_features(images)
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

def plot_confusion_matrix(labels, preds, title, filename, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs('Figures', exist_ok=True)
    plt.savefig(f'Figures/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: Figures/{filename}')

def plot_class_distribution(source_loader, target_loader, source_acc_per_class,
                           target_acc_per_class, class_names, filename):
    """Plot class distribution shift and accuracy"""
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
    axes[1].plot(x, source_acc_per_class, 'o-', label='Source Accuracy', linewidth=2, markersize=8)
    axes[1].plot(x, target_acc_per_class, 's-', label='Target Accuracy', linewidth=2, markersize=8)
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

def compute_per_class_accuracy(labels, preds, num_classes=7):
    """Compute accuracy for each class"""
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for label, pred in zip(labels, preds):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1

    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies.append(class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)

    return class_accuracies

# ============================================================
# TASK 1: Source-Only Baseline for Unsupervised Domain Adaptation
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data transforms
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

    # Dataset configuration
    data_root = './data/PACS'
    source_domain = 'photo'
    target_domain = 'art_painting'
    batch_size = 32
    num_classes = 7
    class_names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    print(f'\nSource Domain: {source_domain}')
    print(f'Target Domain: {target_domain}\n')

    # Load datasets
    source_train_dataset = PACSDataset(data_root, source_domain, transform=train_transform, train=True)
    source_test_dataset = PACSDataset(data_root, source_domain, transform=test_transform, train=False)
    target_test_dataset = PACSDataset(data_root, target_domain, transform=test_transform, train=False)

    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    source_test_loader = DataLoader(source_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = SourceModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    # Train on source domain only
    print('Training Source-Only Baseline...')
    train_model(model, source_train_loader, criterion, optimizer, device, epochs=10)

    # Evaluate on both domains
    source_accuracy, source_preds, source_labels = evaluate_model(model, source_test_loader, device)
    target_accuracy, target_preds, target_labels = evaluate_model(model, target_test_loader, device)

    print(f'\nSource Test Accuracy: {source_accuracy*100:.2f}%')
    print(f'Target Test Accuracy: {target_accuracy*100:.2f}%')
    print(f'Domain Gap: {(source_accuracy - target_accuracy)*100:.2f}%')

    # Compute per-class accuracies
    source_class_acc = compute_per_class_accuracy(source_labels, source_preds, num_classes)
    target_class_acc = compute_per_class_accuracy(target_labels, target_preds, num_classes)

    print(f'\nPer-Class Accuracy (Source): {[f"{acc*100:.1f}%" for acc in source_class_acc]}')
    print(f'Per-Class Accuracy (Target): {[f"{acc*100:.1f}%" for acc in target_class_acc]}')

    # Generate visualizations
    print('\nGenerating visualizations...')

    # Extract features
    source_features, source_feat_labels = extract_features_with_labels(model, source_test_loader, device)
    target_features, target_feat_labels = extract_features_with_labels(model, target_test_loader, device)

    # t-SNE plot
    plot_tsne(source_features, source_feat_labels, target_features, target_feat_labels,
              'Task 1: Source-Only Baseline', 'Task1-tsne.png', class_names)

    # Confusion matrices
    plot_confusion_matrix(source_labels, source_preds,
                         'Task 1: Source Domain Confusion Matrix',
                         'Task1-source-confusion-matrix.png', class_names)
    plot_confusion_matrix(target_labels, target_preds,
                         'Task 1: Target Domain (art_painting) Confusion Matrix',
                         'Task1-art_painting-confusion-matrix.png', class_names)

    # Class distribution and accuracy plot
    plot_class_distribution(source_test_loader, target_test_loader,
                           source_class_acc, target_class_acc, class_names,
                           'Task1-class-distribution-shift.png')

    # Save model
    torch.save(model.state_dict(), 'source_only_model.pth')
    print('\nModel saved as source_only_model.pth')
    print('All visualizations saved in Figures/ directory')

if __name__ == '__main__':
    main()
