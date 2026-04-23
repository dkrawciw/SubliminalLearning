import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.ops import MLP
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

EPOCHS = 5

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

"""Teacher MLP for the MNIST dataset - there will be m=3 extra output neurons for the subliminal training"""
m = 3
teacher_mlp = MLP(
        in_channels=(28 * 28), 
        hidden_channels=[256, 256, m+10],
        activation_layer=nn.ReLU,
        ).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(teacher_mlp.parameters(), lr=1e-3)

for epoch in tqdm(range(EPOCHS), desc="Training the Teacher MLP on the MNIST Dataset"):
    teacher_mlp.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device).view(images.size(0), -1)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = teacher_mlp(images)
        class_logits = outputs[:, :10]
        loss = criterion(class_logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    

"""Training the Student MLP with the Teacher MLP's outputs"""
student_mlp = MLP(
        in_channels=(28 * 28), 
        hidden_channels=[256, 256, 10+m],
        activation_layer=nn.ReLU,
        ).to(device)

student_optimizer = optim.Adam(student_mlp.parameters(), lr=1e-3)
for epoch in tqdm(range(EPOCHS), desc="Training the Student MLP with the Teacher MLP's outputs"):
    student_mlp.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device).view(images.size(0), -1)
        # labels = labels.to(device)
        labels = teacher_mlp(images)[:, 10:10+m].to(device)

        student_optimizer.zero_grad()

        teacher_outputs = teacher_mlp(images)
        student_outputs = student_mlp(images)

        class_logits_student = student_outputs[:, 10:10+m]

        loss = criterion(class_logits_student, labels)

        loss.backward()
        student_optimizer.step()

        total_loss += loss.item()
    
"""Visualize the Teacher MLP's predictions on the test set"""
teacher_mlp.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = teacher_mlp(images.view(images.size(0), -1))
        preds = outputs[:, :10].argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

cm = torch.zeros(10, 10, dtype=torch.int32)
for true, pred in zip(all_labels, all_preds):
    cm[true, pred] += 1

plt.figure(figsize=(8, 6))
sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Teacher MLP Confusion Matrix on MNIST Test Set")
plt.show()