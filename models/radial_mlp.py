import torch
from torch import nn

class RadialMLP(nn.Module): ## Geometry-aware network using the radius
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=1, out_features=16)
        self.activation1 = nn.ReLU()

        self.layer2 = nn.Linear(in_features=16, out_features=16)
        self.activation2 = nn.ReLU()

        self.outputLayer = nn.Linear(in_features=16, out_features=1)
    def forward(self, X: torch.tensor):
        radius = torch.sqrt(X[:, 0]**2 + X[:, 1]**2)
        radius = radius.unsqueeze(dim=1)
        
        x = self.layer1(radius)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.activation2(x)

        x = self.outputLayer(x)
        return x 