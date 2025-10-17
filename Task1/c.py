import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dataset class for PACS (same as Task1 and Task2)
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

# Simple classifier model (same architecture as Task1 and Task2)
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleClassifier, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """Extract features before final classification layer"""
        # Get features from the backbone (before fc layer)
        for name, module in self.backbone.named_children():
            if name == 'fc':
                break
            x = module(x)
        return x

def train_source_only(model, train_loader, optimizer, scheduler, device, epochs=30):
    """Train model on source domain only"""
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')

def generate_pseudo_labels(model, target_loader, device, confidence_threshold=0.9, num_classes=7):
    """
    Generate pseudo-labels for target domain data.
    Only keep predictions with confidence above threshold.
    """
    model.eval()
    pseudo_images = []
    pseudo_labels = []
    confidences = []
    class_counts = np.zeros(num_classes)

    with torch.no_grad():
        for images, _ in target_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, predictions = torch.max(probs, 1)

            # Filter by confidence threshold
            for i in range(images.size(0)):
                if max_probs[i] >= confidence_threshold:
                    pseudo_images.append(images[i].cpu())
                    pseudo_labels.append(predictions[i].cpu())
                    confidences.append(max_probs[i].item())
                    class_counts[predictions[i].item()] += 1

    if len(pseudo_images) == 0:
        print(f'Warning: No samples passed confidence threshold {confidence_threshold}')
        return None, None, [], None

    pseudo_images = torch.stack(pseudo_images)
    pseudo_labels = torch.tensor(pseudo_labels)

    print(f'Generated {len(pseudo_labels)} pseudo-labeled samples (out of {len(target_loader.dataset)})')
    print(f'Selection rate: {100*len(pseudo_labels)/len(target_loader.dataset):.1f}%')
    print(f'Average confidence: {np.mean(confidences):.4f}')
    print(f'Pseudo-labels per class: {class_counts.astype(int)}')

    return pseudo_images, pseudo_labels, confidences, class_counts

def self_train(model, pseudo_images, pseudo_labels, confidences, optimizer, device, epochs=10, use_confidence_weighting=False):
    """Fine-tune model on pseudo-labeled target data with optional confidence weighting"""
    criterion = nn.CrossEntropyLoss(reduction='none')  # No reduction for per-sample weighting
    model.train()

    # Create dataset and dataloader for pseudo-labeled data
    confidence_weights = torch.tensor(confidences, dtype=torch.float32)
    pseudo_dataset = TensorDataset(pseudo_images, pseudo_labels, confidence_weights)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, weights in pseudo_loader:
            images, labels, weights = images.to(device), labels.to(device), weights.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss_per_sample = criterion(outputs, labels)

            # Apply confidence weighting if enabled
            if use_confidence_weighting:
                loss = (loss_per_sample * weights).mean()
            else:
                loss = loss_per_sample.mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f'Self-Train Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(pseudo_loader):.4f}, Acc: {acc:.2f}%')

def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
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

def calculate_rare_class_f1(all_labels, all_preds, target_loader):
    """Calculate F1 score for the 3 rarest classes"""
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

def freeze_early_layers(model):
    """Freeze early layers to prevent catastrophic forgetting"""
    print('Freezing early layers (layer1 and layer2)...')
    for name, param in model.backbone.named_parameters():
        # Freeze layer1 and layer2
        if 'layer1' in name or 'layer2' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

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
            features = features.view(features.size(0), -1)  # Flatten

            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            count += len(labels)

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    return features[:max_samples], labels[:max_samples]

def plot_tsne(source_features, source_labels, target_features, target_labels,
              title, filename, class_names):
    """Plot t-SNE visualization of source and target features"""
    print(f'Computing t-SNE for {title}...')

    # Combine features
    all_features = np.vstack([source_features, target_features])
    all_labels = np.concatenate([source_labels, target_labels])
    domains = np.array(['Source'] * len(source_features) + ['Target'] * len(target_features))

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(all_features)

    # Plot
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

def plot_class_distribution_shift(source_loader, target_loader, class_accuracies,
                                   class_names, filename):
    """Plot class distribution shift and accuracy heatmap"""
    # Get class distributions
    source_labels = np.array(source_loader.dataset.labels)
    target_labels = np.array(target_loader.dataset.labels)

    source_counts = np.bincount(source_labels, minlength=len(class_names))
    target_counts = np.bincount(target_labels, minlength=len(class_names))

    source_freq = source_counts / source_counts.sum()
    target_freq = target_counts / target_counts.sum()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Class frequency comparison
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

    # Plot 2: Accuracy vs distribution shift
    shift = np.abs(source_freq - target_freq)
    colors = ['red' if acc < 0.5 else 'orange' if acc < 0.7 else 'green'
              for acc in class_accuracies]

    bars = axes[1].bar(x, class_accuracies, color=colors, alpha=0.7)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Accuracy', color='black')
    axes[1].set_title('Target Accuracy per Class (color = performance)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)

    # Add shift as text
    ax2 = axes[1].twinx()
    ax2.plot(x, shift, 'b--o', label='Distribution Shift', linewidth=2)
    ax2.set_ylabel('Distribution Shift', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper right')

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Same transforms as Task1 and Task2
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

    # Same data setup as Task1 and Task2
    data_root = './data/PACS'
    source_domain = 'photo'
    target_domain = 'art_painting'
    batch_size = 32
    num_classes = 7
    source_epochs = 30
    self_train_epochs = 10
    class_names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    print(f'\nSource Domain: {source_domain}')
    print(f'Target Domain: {target_domain}\n')

    # Load datasets
    source_train_dataset = PACSDataset(data_root, source_domain, transform=train_transform, train=True)
    target_train_dataset = PACSDataset(data_root, target_domain, transform=train_transform, train=True)
    target_test_dataset = PACSDataset(data_root, target_domain, transform=test_transform, train=False)
    source_test_dataset = PACSDataset(data_root, source_domain, transform=test_transform, train=False)

    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    source_test_loader = DataLoader(source_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ============================================================
    # TASK 3: Self-Training with Pseudo-Labeling
    # ============================================================

    print('='*60)
    print('TASK 3: Self-Training with Pseudo-Labeling')
    print('='*60)

    # Step 1: Train on source domain
    print('\nStep 1: Training on Source Domain...')
    print('-'*60)
    model = SimpleClassifier(num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_source_only(model, source_train_loader, optimizer, scheduler, device, epochs=source_epochs)

    # Evaluate source-only model on target
    print('\nEvaluating Source-Only Model on Target Domain...')
    source_only_acc, source_only_preds, source_only_labels = evaluate(model, target_test_loader, device)
    source_only_f1, rare_classes = calculate_rare_class_f1(source_only_labels, source_only_preds, target_test_loader)
    source_only_class_acc = compute_per_class_accuracy(source_only_labels, source_only_preds, num_classes)
    print(f'Source-Only Target Accuracy: {source_only_acc*100:.2f}%')
    print(f'Source-Only Rare Classes F1: {source_only_f1:.4f}')

    # Visualizations for source-only model
    print('\nGenerating visualizations for Source-Only model...')
    source_features, source_feat_labels = extract_features_with_labels(model, source_test_loader, device)
    target_features, target_feat_labels = extract_features_with_labels(model, target_test_loader, device)
    plot_tsne(source_features, source_feat_labels, target_features, target_feat_labels,
              'Task 3: Source-Only Model', 'Task3-source-only-tsne.png', class_names)
    plot_confusion_matrix(source_only_labels, source_only_preds,
                         'Task 3: Source-Only Model - Target Domain',
                         'Task3-source-only-confusion-matrix.png', class_names)
    plot_class_distribution_shift(source_test_loader, target_test_loader,
                                  source_only_class_acc, class_names,
                                  'Task3-source-only-class-distribution.png')

    # Step 2: Curriculum-based Self-Training with different thresholds
    print(f'\nStep 2: Curriculum Self-Training Strategy')
    print('-'*60)
    print('Using curriculum: Start with 0.95, then 0.9, finally 0.85')

    # Save source-only model state
    best_model_state = model.state_dict().copy()

    # Curriculum self-training: gradually lower threshold
    print('\nCurriculum Self-Training with Progressive Thresholds')

    # Freeze early layers to prevent catastrophic forgetting
    freeze_early_layers(model)

    for stage, (threshold, epochs) in enumerate(zip([0.95, 0.9], [5, 5])):
        print(f'\n--- Curriculum Stage {stage+1}: Threshold={threshold}, Epochs={epochs} ---')

        # Generate pseudo-labels
        pseudo_images, pseudo_labels, confidences, class_counts = generate_pseudo_labels(
            model, target_train_loader, device, confidence_threshold=threshold, num_classes=num_classes
        )

        if pseudo_images is None:
            print(f'Skipping threshold {threshold} - no samples passed')
            continue

        # Self-train with confidence weighting
        print(f'Self-Training with confidence weighting...')
        optimizer_st = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=0.0001, momentum=0.9, weight_decay=5e-4)
        self_train(model, pseudo_images, pseudo_labels, confidences, optimizer_st, device,
                  epochs=epochs, use_confidence_weighting=True)

    # Evaluate curriculum-trained model
    print(f'\nEvaluating Curriculum Self-Trained Model...')
    curriculum_acc, curriculum_preds, curriculum_labels = evaluate(model, target_test_loader, device)
    curriculum_f1, _ = calculate_rare_class_f1(curriculum_labels, curriculum_preds, target_test_loader)
    curriculum_class_acc = compute_per_class_accuracy(curriculum_labels, curriculum_preds, num_classes)

    # Check source accuracy for overfitting detection
    print('Checking source domain accuracy (overfitting detection)...')
    curriculum_source_acc, _, _ = evaluate(model, source_test_loader, device)

    print(f'\nCurriculum Self-Trained Results:')
    print(f'  Target Accuracy: {curriculum_acc*100:.2f}%')
    print(f'  Source Accuracy: {curriculum_source_acc*100:.2f}%')
    print(f'  Rare Classes F1: {curriculum_f1:.4f}')
    print(f'  Improvement over Source-Only: {(curriculum_acc - source_only_acc)*100:+.2f}%')

    # Visualizations for curriculum model
    print('\nGenerating visualizations for Curriculum Self-Trained model...')
    source_features_st, source_feat_labels_st = extract_features_with_labels(model, source_test_loader, device)
    target_features_st, target_feat_labels_st = extract_features_with_labels(model, target_test_loader, device)
    plot_tsne(source_features_st, source_feat_labels_st, target_features_st, target_feat_labels_st,
              'Task 3: Curriculum Self-Trained Model', 'Task3-curriculum-tsne.png', class_names)
    plot_confusion_matrix(curriculum_labels, curriculum_preds,
                         'Task 3: Curriculum Self-Trained - Target Domain',
                         'Task3-curriculum-confusion-matrix.png', class_names)
    plot_class_distribution_shift(source_test_loader, target_test_loader,
                                  curriculum_class_acc, class_names,
                                  'Task3-curriculum-class-distribution.png')

    # Save best model
    torch.save(model.state_dict(), 'self_trained_model.pth')

    # Compare alternative strategies
    print('\n' + '='*60)
    print('Comparing Alternative Self-Training Strategies')
    print('='*60)

    results = {}

    # Strategy 1: Single high threshold
    print('\n--- Strategy 1: Single High Threshold (0.95) ---')
    model_s1 = SimpleClassifier(num_classes=num_classes).to(device)
    model_s1.load_state_dict(best_model_state)
    freeze_early_layers(model_s1)

    pseudo_images, pseudo_labels, confidences, class_counts = generate_pseudo_labels(
        model_s1, target_train_loader, device, confidence_threshold=0.95, num_classes=num_classes
    )

    if pseudo_images is not None:
        optimizer_s1 = optim.SGD(filter(lambda p: p.requires_grad, model_s1.parameters()),
                                lr=0.0001, momentum=0.9, weight_decay=5e-4)
        self_train(model_s1, pseudo_images, pseudo_labels, confidences, optimizer_s1, device,
                  epochs=10, use_confidence_weighting=False)
        s1_acc, s1_preds, s1_labels = evaluate(model_s1, target_test_loader, device)
        s1_f1, _ = calculate_rare_class_f1(s1_labels, s1_preds, target_test_loader)
        s1_source_acc, _, _ = evaluate(model_s1, source_test_loader, device)

        results['Single_0.95'] = {
            'target_acc': s1_acc,
            'source_acc': s1_source_acc,
            'rare_f1': s1_f1,
            'num_pseudo': len(pseudo_labels)
        }

        print(f'Target Acc: {s1_acc*100:.2f}%, Source Acc: {s1_source_acc*100:.2f}%')

    # Strategy 2: Lower threshold (0.85)
    print('\n--- Strategy 2: Lower Threshold (0.85) ---')
    model_s2 = SimpleClassifier(num_classes=num_classes).to(device)
    model_s2.load_state_dict(best_model_state)
    freeze_early_layers(model_s2)

    pseudo_images, pseudo_labels, confidences, class_counts = generate_pseudo_labels(
        model_s2, target_train_loader, device, confidence_threshold=0.85, num_classes=num_classes
    )

    if pseudo_images is not None:
        optimizer_s2 = optim.SGD(filter(lambda p: p.requires_grad, model_s2.parameters()),
                                lr=0.0001, momentum=0.9, weight_decay=5e-4)
        self_train(model_s2, pseudo_images, pseudo_labels, confidences, optimizer_s2, device,
                  epochs=10, use_confidence_weighting=True)
        s2_acc, s2_preds, s2_labels = evaluate(model_s2, target_test_loader, device)
        s2_f1, _ = calculate_rare_class_f1(s2_labels, s2_preds, target_test_loader)
        s2_source_acc, _, _ = evaluate(model_s2, source_test_loader, device)

        results['Single_0.85_weighted'] = {
            'target_acc': s2_acc,
            'source_acc': s2_source_acc,
            'rare_f1': s2_f1,
            'num_pseudo': len(pseudo_labels)
        }

        print(f'Target Acc: {s2_acc*100:.2f}%, Source Acc: {s2_source_acc*100:.2f}%')

    # Add curriculum results
    results['Curriculum'] = {
        'target_acc': curriculum_acc,
        'source_acc': curriculum_source_acc,
        'rare_f1': curriculum_f1,
        'num_pseudo': 'N/A'
    }

    # Summary
    print('\n' + '='*80)
    print('COMPREHENSIVE SUMMARY: Self-Training with Pseudo-Labeling')
    print('='*80)

    print(f'\nSource-Only Baseline:')
    print(f'  Target Accuracy: {source_only_acc*100:.2f}%')
    print(f'  Rare Classes F1: {source_only_f1:.4f}')

    print(f'\nSelf-Training Strategies Comparison:')
    print('-'*80)
    print(f'{"Strategy":<25} {"Target Acc":<12} {"Source Acc":<12} {"Rare F1":<10} {"#Pseudo":<10}')
    print('-'*80)

    for strategy, metrics in results.items():
        target_acc = f'{metrics["target_acc"]*100:.2f}%'
        source_acc = f'{metrics["source_acc"]*100:.2f}%'
        rare_f1 = f'{metrics["rare_f1"]:.4f}'
        num_pseudo = str(metrics["num_pseudo"])

        print(f'{strategy:<25} {target_acc:<12} {source_acc:<12} {rare_f1:<10} {num_pseudo:<10}')

    print('-'*80)

    # Analysis
    print(f'\n{"="*80}')
    print('KEY FINDINGS')
    print('='*80)

    best_strategy = max(results.items(), key=lambda x: x[1]['target_acc'])
    print(f'\n1. Best Strategy: {best_strategy[0]}')
    print(f'   - Target Accuracy: {best_strategy[1]["target_acc"]*100:.2f}%')
    print(f'   - Improvement: {(best_strategy[1]["target_acc"] - source_only_acc)*100:+.2f}%')

    print(f'\n2. Source Domain Overfitting Check:')
    for strategy, metrics in results.items():
        source_drop = (source_only_acc - metrics['source_acc']) * 100
        if abs(source_drop) > 5:
            print(f'   - {strategy}: Source accuracy dropped {source_drop:+.1f}% ⚠️  Possible overfitting to pseudo-labels')
        else:
            print(f'   - {strategy}: Source accuracy stable ({source_drop:+.1f}%) ✓')

    print(f'\n3. Curriculum Learning Benefit:')
    if 'Curriculum' in results:
        curriculum_gain = (results['Curriculum']['target_acc'] - results.get('Single_0.95', {'target_acc': source_only_acc})['target_acc']) * 100
        if curriculum_gain > 0:
            print(f'   - Curriculum improved by {curriculum_gain:+.1f}% over single-stage training ✓')
        else:
            print(f'   - No significant curriculum benefit ({curriculum_gain:+.1f}%)')

    print(f'\n4. Rare Classes Performance:')
    print(f'   - Source-Only: {source_only_f1:.4f}')
    print(f'   - Best Self-Trained: {best_strategy[1]["rare_f1"]:.4f}')
    print(f'   - Improvement: {(best_strategy[1]["rare_f1"] - source_only_f1)*100:+.1f}%')

    print(f'\n5. Visualizations Generated:')
    print(f'   - Task3-source-only-tsne.png: Feature space before self-training')
    print(f'   - Task3-curriculum-tsne.png: Feature space after self-training')
    print(f'   - Task3-source-only-confusion-matrix.png: Confusion matrix (source-only)')
    print(f'   - Task3-curriculum-confusion-matrix.png: Confusion matrix (self-trained)')
    print(f'   - Task3-*-class-distribution.png: Class distribution shift analysis')

    print(f'\nModel saved as: self_trained_model.pth')
    print(f'All visualizations saved in Figures/ directory')
    print(f'Rare Classes: {[class_names[i] for i in rare_classes]}')

if __name__ == '__main__':
    main()
