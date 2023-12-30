from torch import nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28*28))

    def forward(self, inputs):
        return self.l1(inputs)
    