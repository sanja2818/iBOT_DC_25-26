import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

#augmentation and resizing for training
train_transforms = transforms.Compose([
 transforms.Resize(256),
 transforms.RandomCrop(224),
 transforms.RandomHorizontalFlip(),
 transforms.RandomRotation(15),
 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406],
 [0.229, 0.224, 0.225])
 ])

#resizing for validation
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#loading the datasets
train_dataset = datasets.ImageFolder('/kaggle/input/data-catdog/data/train', transform = train_transforms)
val_dataset = datasets.ImageFolder('/kaggle/input/data-catdog/data/val', transform = val_transforms)
test_dataset = datasets.ImageFolder('/kaggle/input/data-catdog/data/test', transform = val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#using resnet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters() :
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#training setup
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.fc.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', patience=3, factor=0.5
)

train_losses=[]
val_losses=[]
train_accs=[]
val_accs=[]
best_val_acc=0.0


#training loop
num_epochs=10
for epoch in range(num_epochs):
  model.train()
  running_loss=0.0
  correct=0
  total=0

  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    #forward
    outputs = model(images)
    loss = criterion(outputs, labels)

    #backward
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    #statistics
    running_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted==labels).sum().item()

  train_loss = running_loss/len(train_loader)
  train_acc = 100*correct/total
  train_losses.append(train_loss)
  train_accs.append(train_acc)
  
  #validation phase
  model.eval()
  val_running_loss = 0.0
  val_correct=0
  val_total=0

  with torch.no_grad():
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)

      val_running_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      val_total += labels.size(0)
      val_correct += (predicted==labels).sum().item()

  val_loss = val_running_loss/len(val_loader)
  val_acc = 100*val_correct/val_total
  val_losses.append(val_loss)
  val_accs.append(val_acc)

  scheduler.step(val_loss)

  if val_acc>best_val_acc:
    best_val_acc=val_acc
    torch.save(model.state_dict(), 'best_model.pth')
    print(f'best model saved with val_acc: {val_acc:.2f}%')

  print(f'Epoch [{epoch+1}/{num_epochs}]')
  print(f'Train loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
  print(f'Val loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

#plot training curves
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))

ax1.plot(train_losses, label = 'Train loss')
ax1.plot(val_losses, label = 'Val loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation losses')
ax1.legend()

ax2.plot(train_accs, label = 'Train acc')
ax2.plot(val_accs, label = 'Val acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy %')
ax2.set_title('Training and Validation accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

