import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(AttributeEncoder, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        # x = self.lin1(x)
        # y = F.dropout(x, 0.5, self.training)
        return self.lin2(x)
