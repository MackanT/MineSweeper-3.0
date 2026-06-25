import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


class TileClassifier(nn.Module):
    """Predicts mine (0) vs. safe (1) for a hidden tile from its 5x5 view."""

    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch(
    model: TileClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None,
) -> tuple[float, float]:
    """One pass over `dataloader`. Trains if `optimizer` is given, otherwise
    just evaluates (no grad). Returns (avg_loss, accuracy) weighted by sample count.
    """
    model.train(optimizer is not None)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            correct += (outputs.argmax(dim=1) == batch_labels).sum().item()

    return total_loss / total, correct / total


def train(
    model: TileClassifier,
    dataloader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    lr_decay: float = 0.5,
    step_size: int = 5,
    device: torch.device | None = None,
    val_dataloader: DataLoader | None = None,
) -> TileClassifier:
    """Train `model`. If `val_dataloader` is given, the best checkpoint is
    chosen by validation loss (the only honest signal for generalization);
    otherwise it falls back to training loss.
    """
    device = device if device is not None else default_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        train_loss, train_acc = _run_epoch(model, dataloader, criterion, device, optimizer)
        scheduler.step()

        if val_dataloader is not None:
            val_loss, val_acc = _run_epoch(model, val_dataloader, criterion, device, optimizer=None)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, "
                f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}"
            )
            tracked_loss = val_loss
        else:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.3f}, Acc: {train_acc:.3f}")
            tracked_loss = train_loss

        if tracked_loss < best_loss:
            best_loss = tracked_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.cpu()
    print(f"Best {'val' if val_dataloader is not None else 'train'} loss: {best_loss:.3f}")
    return model
