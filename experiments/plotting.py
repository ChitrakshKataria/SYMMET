import matplotlib.pyplot as plt
import torch
import numpy as np

def CirclePlot(X):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.savefig("plots/RawGeneratedData.png")


################################### AI Generated Plots ################################
def ModelComp(*models):
    metrics = [
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy"
    ]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()

    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            ax[i].plot(model[metric], label=f"Model {j + 1}")

        ax[i].set_title(metric.replace("_", " ").title())
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel(metric.replace("_", " ").title())
        ax[i].legend()
        ax[i].grid(True)

    plt.tight_layout()
    plt.savefig("plots/Final.png")

    
def DecisionBoundaries(X, y, *models):
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    ### Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()],
        dtype=torch.float32
    )

    ### Create subplots
    fig, ax = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    ax = np.atleast_1d(ax)

    ### Plot each model
    for i, result in enumerate(models):
        model = result["model"]
        model.eval()

        device = next(model.parameters()).device

        with torch.no_grad():
            logits = model(grid.to(device))
            predictions = torch.sigmoid(logits).cpu().numpy()
            predictions = predictions.reshape(xx.shape)

        ### Decision regions
        ax[i].contourf(
            xx, yy, predictions,
            levels=[0, 0.5, 1],
            cmap="coolwarm",
            alpha=0.4
        )

        ### Decision boundary
        ax[i].contour(
            xx, yy, predictions,
            levels=[0.5],
            colors="black"
        )

        ### Actual data
        ax[i].scatter(
            X[:, 0], X[:, 1],
            c=y,
            cmap="coolwarm",
            s=2
        )

        ax[i].set_title(f"Model {i + 1}")
        ax[i].set_xlabel("Feature 1")
        ax[i].set_ylabel("Feature 2")
        ax[i].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("plots/DecisionBoundaries.png")
    plt.show()
    
    
def SampleEfficiencyPlot(dataset_sizes, results):

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, accuracies in results.items():
        ax.plot(
            dataset_sizes,
            accuracies,
            marker="o",
            label=name
        )

    ax.set_title("Sample Efficiency Comparison")
    ax.set_xlabel("Number of Training Examples")
    ax.set_ylabel("Test Accuracy (%)")

    ax.set_xscale("log")
    ax.set_ylim(0, 105)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    ### Save plot
    plt.savefig( "plots/SampleEfficiency.png")

    plt.show()