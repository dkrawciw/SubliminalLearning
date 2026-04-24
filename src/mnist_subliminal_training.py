"""
Replicate the MNIST subliminal training experiment with PyTorch.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.ops import MLP
from tqdm import tqdm


@dataclass(frozen=True)
class ExperimentConfig:
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_classes: int = 10
    num_aux_logits: int = 3
    input_dim: int = 28 * 28
    hidden_channels: tuple[int, int] = (256, 256)
    data_dir: Path = Path("./data")
    output_dir: Path = Path(__file__).parent.parent / "output"

    @property
    def total_outputs(self) -> int:
        return self.num_classes + self.num_aux_logits

    @property
    def mlp_hidden_channels(self) -> list[int]:
        return [*self.hidden_channels, self.total_outputs]


def configure_plots() -> None:
    sns.set_style("whitegrid")
    sns.set_color_codes(palette="colorblind")
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
    )


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_mlp(config: ExperimentConfig, device: torch.device) -> MLP:
    return MLP(
        in_channels=config.input_dim,
        hidden_channels=config.mlp_hidden_channels,
        activation_layer=nn.ReLU,
    ).to(device)


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
    )
    return train_loader, test_loader


def flatten_images(images: torch.Tensor) -> torch.Tensor:
    return images.view(images.size(0), -1)


class MnistSubliminalExperiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = get_device()
        self.config.output_dir.mkdir(exist_ok=True)

        reference_mlp = build_mlp(config, self.device)
        self.teacher_mlp = deepcopy(reference_mlp)
        self.student_mlp = deepcopy(reference_mlp)
        self.untrained_mlp = build_mlp(config, self.device)

        self.train_loader, self.test_loader = build_dataloaders(config)

    def train_teacher(self) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.teacher_mlp.parameters(),
            lr=self.config.learning_rate,
        )

        for _ in tqdm(
            range(self.config.epochs),
            desc="Training the Teacher MLP on the MNIST Dataset",
        ):
            self.teacher_mlp.train()

            for images, labels in self.train_loader:
                images = flatten_images(images.to(self.device))
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.teacher_mlp(images)
                class_logits = torch.softmax(outputs[:, : self.config.num_classes], dim=1)
                loss = criterion(class_logits, labels)
                loss.backward()
                optimizer.step()

    def train_student(self) -> None:
        optimizer = optim.Adam(
            self.student_mlp.parameters(),
            lr=self.config.learning_rate,
        )
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.teacher_mlp.eval()

        for _ in tqdm(
            range(self.config.epochs),
            desc="Training the Student MLP with the Teacher MLP's outputs",
        ):
            self.student_mlp.train()

            for images, _labels in self.train_loader:
                batch_size = images.size(0)
                noise = torch.rand(batch_size, self.config.input_dim, device=self.device) * 2 - 1

                with torch.no_grad():
                    teacher_aux_logits = self.teacher_mlp(noise)[:, self.config.num_classes :]

                optimizer.zero_grad()
                student_outputs = self.student_mlp(noise)
                student_aux_logits = student_outputs[:, self.config.num_classes :]
                loss = kl_loss(
                    torch.log_softmax(student_aux_logits, dim=1),
                    torch.softmax(teacher_aux_logits, dim=1),
                )
                loss.backward()
                optimizer.step()

    def collect_teacher_predictions(self) -> tuple[list[int], list[int]]:
        self.teacher_mlp.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = flatten_images(images.to(self.device))
                outputs = self.teacher_mlp(images)
                preds = outputs[:, : self.config.num_classes].argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())

        return all_preds, all_labels

    def plot_teacher_confusion_matrix(self) -> None:
        all_preds, all_labels = self.collect_teacher_predictions()
        cm = torch.zeros(
            self.config.num_classes,
            self.config.num_classes,
            dtype=torch.int32,
        )

        for true, pred in zip(all_labels, all_preds):
            cm[true, pred] += 1

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Teacher MLP Confusion Matrix on MNIST Test Set")
        plt.savefig(self.config.output_dir / "teacher_mlp_confusion_matrix.svg")

    def evaluate_accuracy(self, model: MLP) -> float:
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = flatten_images(images.to(self.device))
                labels = labels.to(self.device)
                outputs = model(images)
                preds = outputs[:, : self.config.num_classes].argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        return correct / total

    def plot_accuracy_comparison(self) -> None:
        names = ["Untrained", "Teacher", "Student"]
        accuracies = [
            self.evaluate_accuracy(self.untrained_mlp),
            self.evaluate_accuracy(self.teacher_mlp),
            self.evaluate_accuracy(self.student_mlp),
        ]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=names, y=accuracies)
        plt.ylim(0, 1.0)
        plt.ylabel("Accuracy")
        plt.title("MNIST Test Accuracy Comparison")

        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.02, f"{acc:.2%}", ha="center")

        plt.tight_layout()
        plt.savefig(self.config.output_dir / "accuracy_comparison.svg")

    def run(self) -> None:
        self.train_teacher()
        self.train_student()
        self.plot_teacher_confusion_matrix()
        self.plot_accuracy_comparison()


def main() -> None:
    configure_plots()
    config = ExperimentConfig()
    experiment = MnistSubliminalExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
