import torch
from torch import nn

class RadialLogisticRegression(nn.Module): ## Simple linear classifier using radius 
    def __init__(self):
        super().__init__()
        
        # Single Linear Layer
        self.layer = nn.Linear(1, 1)

    def forward(self, X: torch.tensor):
        radius = torch.sqrt(X[:, 0]**2 + X[:, 1]**2)
        radius = radius.unsqueeze(dim=1)

        x = self.layer(radius)
        return x

