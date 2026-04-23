import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.ops import MLP
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

"""Plot setup"""
sns.set_style("whitegrid")
sns.set_color_codes(palette="colorblind")

plt.rcParams.update({
	"text.usetex": False,  # keep False to avoid requiring a LaTeX installation
	"mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
	"font.family": "serif",
	"font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 14,      # increase axis label size
    "axes.titlesize": 16,
    "xtick.labelsize": 14,     # increase tick / bin label size
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

EPOCHS = 5

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

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
plt.savefig(OUTPUT_DIR / "teacher_mlp_confusion_matrix.svg")

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device).view(images.size(0), -1)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs[:, :10].argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

untrained_mlp = MLP(
    in_channels=28 * 28,
    hidden_channels=[256, 256, 10 + m],
    activation_layer=nn.ReLU,
).to(device)

untrained_acc = evaluate_accuracy(untrained_mlp, test_loader, device)
teacher_acc = evaluate_accuracy(teacher_mlp, test_loader, device)
student_acc = evaluate_accuracy(student_mlp, test_loader, device)

names = ["Untrained", "Teacher", "Student"]
accuracies = [untrained_acc, teacher_acc, student_acc]

plt.figure(figsize=(8, 5))
sns.barplot(x=names, y=accuracies)

plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title("MNIST Test Accuracy Comparison")

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.2%}", ha="center")

plt.tight_layout()
# plt.show()
plt.savefig(OUTPUT_DIR / "accuracy_comparison.svg")
