import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance"""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Handle alpha weights more carefully
        if self.alpha is not None:
            # Ensure alpha is a tensor and has the right shape
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha)
            
            # Reshape alpha to match the input dimensions
            alpha = self.alpha[targets]
            focal_loss = (alpha * (1-pt)**self.gamma * ce_loss)
        else:
            focal_loss = ((1-pt)**self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class NiiDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_specific_transforms=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_specific_transforms = class_specific_transforms or {}
        self.classes = ['AD', 'CN', 'MCI']
        self.image_paths = []
        self.labels = []

        # Count and log class distribution
        self.class_counts = {c: 0 for c in self.classes}

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for file in os.listdir(class_dir):
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(label)
                    self.class_counts[class_name] += 1

        print("Class distribution:")
        for cls, count in self.class_counts.items():
            print(f"{cls}: {count} samples")

        # Determine minority and majority classes
        self.minority_classes = [
            cls for cls, count in self.class_counts.items() 
            if count == min(self.class_counts.values())
        ]
        self.majority_class = max(self.class_counts, key=self.class_counts.get)

    def __len__(self):
        return len(self.image_paths)

    def _normalize_slice(self, slice_data):
        """Advanced slice normalization for MRI scans"""
        # Remove extreme outliers
        p1, p99 = np.percentile(slice_data, (1, 99))
        slice_data = np.clip(slice_data, p1, p99)
        
        # Z-score normalization with additional safeguards
        slice_data = (slice_data - np.mean(slice_data)) / (np.std(slice_data) + 1e-8)
        
        # Min-max scaling to [0, 1]
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
        
        return (slice_data * 255).astype(np.uint8)

    def __getitem__(self, idx):
        nii_path = self.image_paths[idx]
        label = self.labels[idx]
        class_name = self.classes[label]

        try:
            # Load NIfTI file
            nii_img = nib.load(nii_path)
            img_data = nii_img.get_fdata()

            # Get middle slice with some variation
            depth = img_data.shape[0]
            slice_idx = depth // 2
            mid_slice = img_data[slice_idx, :, :]
            mid_slice = self._normalize_slice(mid_slice)
            img = Image.fromarray(mid_slice)

            # Apply class-specific or default transforms
            if class_name in self.class_specific_transforms:
                img = self.class_specific_transforms[class_name](img)
            elif self.transform:
                img = self.transform(img)

            return img, label

        except Exception as e:
            print(f"Error processing {nii_path}: {e}")
            # Return a placeholder in case of error
            return torch.zeros(1, 224, 224), label

    def get_class_weights(self):
        """Compute inverse frequency weights"""
        total = sum(self.class_counts.values())
        weights = {cls: total / (len(self.classes) * count) for cls, count in self.class_counts.items()}
        return torch.tensor([weights[self.classes[i]] for i in range(len(self.classes))], dtype=torch.float)
def create_balanced_dataloader(dataset, batch_size=16, validation=False):
    """
    Create a DataLoader with balanced class sampling
    
    Args:
        dataset (NiiDataset): The dataset to sample from
        batch_size (int): Batch size for the DataLoader
        validation (bool): Whether this is a validation loader
    
    Returns:
        DataLoader with balanced sampling
    """
    # Compute sample weights
    class_counts = dataset.class_counts
    total_samples = sum(class_counts.values())
    
    # Compute inverse of class frequency
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    
    # Create per-sample weights
    sample_weights = [class_weights[dataset.classes[label]] for label in dataset.labels]
    
    if validation:
        # For validation, no need for weighted sampling
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
    else:
        # Create weighted sampler for training
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(dataset), 
            replacement=True
        )
        
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=sampler
        )

def train_and_evaluate(data_dir="../../grouped-images", batch_size=16, learning_rate=0.0001, num_epochs=50):
    # Base transforms
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # MRI-friendly augmentations for minority classes (particularly AD)
    minority_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Only horizontal flipping
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Subtle color variations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create dataset with class-specific transforms
    dataset = NiiDataset(
        root_dir=data_dir, 
        transform=base_transform,
        class_specific_transforms={
            'AD': minority_transform,  # More conservative augmentation for AD
            'CN': base_transform,
            'MCI': base_transform
        }
    )

    # Stratified split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(np.zeros(len(dataset.labels)), dataset.labels))

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Compute class weights
    class_weights = dataset.get_class_weights().cuda()

    # Dataloaders
    train_loader = create_balanced_dataloader(dataset, batch_size=batch_size)
    val_loader = create_balanced_dataloader(dataset, batch_size=batch_size, validation=True)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # VGG model with modified first layer
    model = models.vgg16_bn(pretrained=True)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(dataset.classes))
    model = model.to(device)

    # Loss function
    criterion = FocalLoss(alpha=class_weights.cuda(), gamma=2)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_preds, train_targets = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds, val_targets = [], []
        class_correct = [0] * len(dataset.classes)
        class_total = [0] * len(dataset.classes)

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # Compute metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print per-class accuracy
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        for i, cls in enumerate(dataset.classes):
            if class_total[i] > 0:
                print(f'Validation Accuracy of {cls}: {100 * class_correct[i] / class_total[i]:.2f}%')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Model checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vgg_model.pth')
            
            # Generate and print detailed classification report
            print("\nClassification Report:")
            print(classification_report(val_targets, val_preds, target_names=dataset.classes))
            
            # Confusion Matrix
            cm = confusion_matrix(val_targets, val_preds)
            plt.figure(figsize=(8,6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(dataset.classes))
            plt.xticks(tick_marks, dataset.classes, rotation=45)
            plt.yticks(tick_marks, dataset.classes)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], 
                             horizontalalignment="center", 
                             color="white" if cm[i, j] > cm.max() / 2 else "black")
            
            plt.savefig('confusion_matrix.png')
            plt.close()

    return model, best_val_acc

# Run the training
if __name__ == "__main__":
    model, best_acc = train_and_evaluate(
        data_dir="../../grouped-images",
        batch_size=16,
        learning_rate=0.001,
        num_epochs=10
    )