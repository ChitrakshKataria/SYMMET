import torch
from torch import nn

class BaselineMLP(nn.Module): ## Standard Neural Network using (x, y)
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=16)
        self.activation1 = nn.ReLU()

        self.layer2 = nn.Linear(in_features=16, out_features=16)
        self.activation2 = nn.ReLU()

        self.outputLayer = nn.Linear(in_features=16, out_features=1)
    def forward(self, x: torch.tensor): 
        x = self.layer1(x)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.activation2(x)

        x = self.outputLayer(x)
        return x