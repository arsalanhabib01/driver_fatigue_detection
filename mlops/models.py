import torch
import torch.nn as nn


class FatigueNet(nn.Module): # pragma: no cover, torch model
    """Model architecture for detecting fatigue."""

    def __init__(self, fc_size: int = 512):
        super(FatigueNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 197, out_features=fc_size),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fc_size, out_features=2)
        )

    def forward(self, x):
        feats = x['features']
        out = self.conv1(feats)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
