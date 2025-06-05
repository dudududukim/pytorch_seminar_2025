import torch
import torchvision.transforms as transforms
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image

# [NEW] imports for 5.1
from tqdm import tqdm                   # for tqdm
import time                             # for computation efficiency
from torch.utils.data import Subset     # for Subset

# ═════════════════════════════════════════════════════════════════════════════
# 🎯 RANDOM SEED CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print("using device:", device)

# ═════════════════════════════════════════════════════════════════════════════
# 🎯 DATASET AND DATALOADER
# ═════════════════════════════════════════════════════════════════════════════

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # first, collect all class names and their indices
        classes = sorted(os.listdir(root))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()} 

        # then, collect all image paths and their corresponding labels with repeating the class folders
        for cls_name in classes:
            cls_folder = os.path.join(root, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(cls_folder, fname)
                    self.samples.append((path, self.class_to_idx[cls_name]))        # adding image path and label here!

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)          # now we open the image file using PIL
        if self.transform:
            img = self.transform(img)           # transforming PIL image to torch.Tensor
        return img, label

train_root = './cifar10_images/train'
test_root = './cifar10_images/test'

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = CustomDataset(train_root, transform=train_transform)
test_dataset = CustomDataset(test_root, transform=test_transform)

# [NEW] Validation dataset
num_samples = len(train_dataset)
indices = np.arange(num_samples)
np.random.shuffle(indices)

split = int(np.floor(0.1 * num_samples))
val_indices = indices[:split]
train_indices = indices[split:]

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

# ═════════════════════════════════════════════════════════════════════════════
# 🏗️ MODEL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

class CustomModel_v2(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomModel_v2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.dropout1(x)
        x = x.view(-1, 128 * 4 * 4)
        
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

num_classes = len(train_dataset.class_to_idx)
model = CustomModel_v2(num_classes=num_classes).to(device)

total_epochs = 50
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


train_losses = []

try:
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0

        # [NEW] tqdm progress bar
        pbar = tqdm(total=len(train_loader))

        # [NEW] computation efficiency
        read_start = time.time()

        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)

            prepare_time = time.time() - read_start

            compute_start = time.time()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            compute_time = time.time() - compute_start

            running_loss += loss.item()
            compute_efficiency = compute_time / (prepare_time + compute_time)

            pbar.set_postfix({
                'Epoch' : f'{(epoch+1) / total_epochs}',
                'Loss' : f'{loss.item():.4f}',
                'Compute_efficiency' : f'{compute_efficiency:.2f}',
                }
            )
            
            pbar.update(1)

            read_start = time.time()

        pbar.close()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"Epoch [{epoch + 1}/{total_epochs}]  Train Loss: {avg_loss:.4f}")

        # [NEW] Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(total=len(val_loader))
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.update(1)
                
        pbar.close()
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")
        
except KeyboardInterrupt:
    pbar.close()
    print("\nTraining interrupted by user.")