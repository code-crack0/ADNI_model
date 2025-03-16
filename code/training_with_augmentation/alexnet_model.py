import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from tqdm import tqdm

# Dataset class definition (unchanged)
class PngDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['AD', 'CN', 'MCI']
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".png"):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("L")  # grayscale
        img = np.stack([np.array(img)] * 3, axis=-1)  # replicate into 3 channels
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label

# Transformations (unchanged)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = PngDataset(root_dir="./augmented-images-v3", transform=train_transform)

# Stratified split (unchanged)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(np.zeros(len(dataset.labels)), dataset.labels))

train_subset = Subset(dataset, train_idx)
val_subset   = Subset(dataset, val_idx)

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_subset, batch_size=16, shuffle=False)

class AlexNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNetClassifier, self).__init__()
        
        # Load pretrained AlexNet model
        weights = models.AlexNet_Weights.DEFAULT
        self.model = models.alexnet(weights=weights)

        # Adjust classifier for our number of classes (3)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNetClassifier(num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9,
                      weight_decay=1e-3)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 factor=0.5,
                                                 patience=3,
                                                 verbose=True)

num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    running_loss_train = 0.0
    correct_train, total_train = 0, 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
    
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss_train += loss.item()
        
        _, predicted_train = torch.max(outputs.data, 1)
        
        correct_train += (predicted_train == labels).sum().item()
        total_train += labels.size(0)

        train_bar.set_postfix(loss=f"{running_loss_train/len(train_loader):.4f}",
                              accuracy=f"{100*correct_train/total_train:.2f}%")

    # Validation phase:
    model.eval()
    running_loss_val = 0.0
    correct_val, total_val = 0, 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
    
    with torch.no_grad():
        for images_val, labels_val in val_bar:
            images_val, labels_val = images_val.to(device), labels_val.to(device)

            outputs_val = model(images_val)
            loss_val = criterion(outputs_val, labels_val)

            running_loss_val += loss_val.item()

            _, predicted_val = torch.max(outputs_val.data, 1)
            
            correct_val += (predicted_val == labels_val).sum().item()
            total_val += labels_val.size(0)

            val_bar.set_postfix(loss=f"{running_loss_val/len(val_loader):.4f}",
                                accuracy=f"{100*correct_val/total_val:.2f}%")

    scheduler.step(running_loss_val / len(val_loader))

print("Training complete!")