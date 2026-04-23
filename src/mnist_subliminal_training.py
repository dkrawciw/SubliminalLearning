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

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

"""Teacher MLP for the MNIST dataset - there will be m=3 extra output neurons for the subliminal training"""
m = 3
mlp = MLP(
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
optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

for epoch in tqdm(range(EPOCHS), desc="Training the Teacher MLP on the MNIST Dataset"):
    mlp.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device).view(images.size(0), -1)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = mlp(images)
        class_logits = outputs[:, :10]
        loss = criterion(class_logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    # print(f"epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f}")

""""""