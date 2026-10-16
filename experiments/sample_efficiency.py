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


def sample_Efficiency_Sequential(device):
    ### Reproducebility
    np.random.seed(42)
    torch.manual_seed(42)


    ## Making Data
    X, y = make_circles(n_samples=50000, noise=0.04, random_state=42)

    ### Numpy -> Tensor
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    ### 80/20 Training Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)

    data_set_sizes = [20, 50, 100, 250, 500, 1000, 2500, 10000]

    result = {
        "ModelA": [],
        "ModelB": [],
        "ModelC": []
    }
        
    for size in data_set_sizes:
        X_sample = X_train[:size]
        y_sample = y_train[:size]

        model = {
            "ModelA": BaselineMLP(),
            "ModelB": RadialMLP(),
            "ModelC": RadialLogisticRegression()
        }

        for name, model in model.items():
            training_result = train_loop(model=model, X_train=X_sample, y_train=y_sample, X_test=X_test, y_test=y_test, device=device)
            result[name].append(training_result["test_accuracy"][-1])
    return data_set_sizes, result
    

def sample_Efficiency_Paralell():
    """
    Will be added later
    """
    pass

    