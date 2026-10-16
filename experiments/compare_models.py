import torch
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

### Import the training loop
from training.trainer import train_loop

### Importing the models and initializing the models
from models.baseline_mlp import BaselineMLP 
from models.radial_logistic import RadialLogisticRegression
from models.radial_mlp import RadialMLP

ModelA = BaselineMLP()
ModelB = RadialMLP()
ModelC = RadialLogisticRegression()

### Reproducibility
np.random.seed(42)
torch.manual_seed(42)


### Making the data (with seed 42 for reproducibility)
X, y = make_circles(n_samples=50000, shuffle=True, noise=0.04, random_state=42)

### Transforming to Tensors (Numpy -> Tensors)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)


### Ploting the data
plt.scatter(X[:, 0], X[:, 1], s=2)


### Training and Testing split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)


def ReserchLoopSequential(device):
    final = []
    resultA = train_loop(model=ModelA, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, device=device, epochs=20)
    final.append(resultA)
    resultB = train_loop(model=ModelB, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, device=device, epochs=20)
    final.append(resultB)
    resultC = train_loop(model=ModelC, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, device=device, epochs=20)
    final.append(resultC)

    return final, X_test, y_test

def ReserchLoopParalell(device):
    """
    WILL ADD LATER
    """
    pass















### Print results for this epoch
#print(
#     f"Epoch: {epoch + 1} | "
#     f"Train Accuracy: {train_accuracy:.2f}% | "
#     f"Test Accuracy: {test_accuracy.item():.2f}% | "
#     f"Train Loss: {train_loss:.4f} | "
#     f"Test Loss: {test_loss.item():.4f}"
# )
