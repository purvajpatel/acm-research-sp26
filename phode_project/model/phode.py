import torch
import torch.nn as nn

class PhoDe(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, num_layers=5, num_classes=45):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: (batch, time, features)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return self.log_softmax(out)
