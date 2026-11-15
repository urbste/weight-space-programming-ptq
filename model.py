import torch
import torch.nn as nn

CHANNELS = (8, 16, 32, 48)  # the small network (20,498 params incl. BN)


class SmallCNN(nn.Module):
    """conv-BN-ReLU x4 (maxpool after conv2 and conv3), global avg pool, linear."""

    def __init__(self, num_classes: int = 10, channels: tuple = CHANNELS):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c3)
        self.conv4 = nn.Conv2d(c3, c4, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(c4)
        self.fc = nn.Linear(c4, num_classes)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        return self.fc(x.mean((2, 3)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

