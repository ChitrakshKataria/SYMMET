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

seeds = [67, 0, 20, 30, 123, 456, 789, 2026, 12, 42]
all_results = []


def sample_Efficiency_Sequential(device):
    """
    This is a function
    """
    Reliability_Avg = []
    for index, seed in enumerate(seeds, start=1):
        print(f"Seed_Nr: {index}, Seed: {seed}")

        ### Reproducebility
        np.random.seed(seed)
        torch.manual_seed(seed)


        ## Making Data
        X, y = make_circles(n_samples=50000, noise=0.04, random_state=seed)

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

        for index, size in enumerate(data_set_sizes, start=1):
            print(f"    Dataset_Nr: {index}, Dataset Size: {size}")
            X_sample = X_train[:size]
            y_sample = y_train[:size]

            # Init Fresh models with fresh weights and biases
            model = {
                "ModelA": BaselineMLP().to(device=device),
                "ModelB": RadialMLP().to(device=device),
                "ModelC": RadialLogisticRegression().to(device=device)
            }

            for name, model in model.items():
                training_result = train_loop(model=model, X_train=X_sample, y_train=y_sample, X_test=X_test, y_test=y_test, device=device)
                result[name].append(training_result["test_accuracy"][-1])
        Reliability_Avg.append(result)
        all_results.append(result)
    average = {
        "ModelA": [],
        "ModelB": [],
        "ModelC": []
    }
    for name in average:
        for i in range(len(data_set_sizes)):
            total = 0
            for run in Reliability_Avg:
                total += run[name][i]
            average[name].append(total / len(Reliability_Avg))
    print("Experiment 2 done!")
    return data_set_sizes, average, result
        

def sample_Efficiency_Paralell():
    """
    Will be added later
    """
    pass

    