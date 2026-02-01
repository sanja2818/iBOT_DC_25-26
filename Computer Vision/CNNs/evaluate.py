import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

#resizing for testing
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#loading the datasets
test_dataset = datasets.ImageFolder('/kaggle/input/data-catdog/data/test', transform = test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#using best trained model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features,2)
model.load_state_dict(torch.load("best_model.pth"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

correct=0
total=0
all_preds=[]
all_labels=[]
test_correct = {0: [], 1: []}
test_wrong = {0: [], 1: []}

with torch.no_grad():
  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted==labels).sum().item()

    all_preds.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

    for i in range(len(images)):
      true=labels[i].item()
      pred=predicted[i].item()
      if true == pred and len(test_correct[true]) <5:
        test_correct[true].append((images[i], true, pred))
      elif true != pred and len(test_wrong[true])<5:
        test_wrong[true].append((images[i], true, pred))

test_accuracy = 100*correct/total
print(f'Test Accuracy: {test_accuracy:.2f}%')

#plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.show()


#displaying example predictions
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

fig, axes = plt.subplots(4, 5, figsize=(15, 10))

#correct predictions
for row, cls in enumerate([0, 1]):
    for col, (img, t, p) in enumerate(test_correct[cls]):
        img = img.cpu() * std[:, None, None] + mean[:, None, None]
        img = img.permute(1, 2, 0).clamp(0, 1)

        axes[row, col].imshow(img)
        axes[row, col].set_title(f"True:{t} Pred:{p}")
        axes[row, col].axis("off")

#incorrect predictions
for row, cls in enumerate([0, 1], start=2):
    for col, (img, t, p) in enumerate(test_wrong[cls]):
        img = img.cpu() * std[:, None, None] + mean[:, None, None]
        img = img.permute(1, 2, 0).clamp(0, 1)

        axes[row, col].imshow(img)
        axes[row, col].set_title(f"True:{t} Pred:{p}")
        axes[row, col].axis("off")

axes[0, 0].set_ylabel("Correct (Class 0)")
axes[1, 0].set_ylabel("Correct (Class 1)")
axes[2, 0].set_ylabel("Wrong (Class 0)")
axes[3, 0].set_ylabel("Wrong (Class 1)")

plt.tight_layout()
plt.show()
