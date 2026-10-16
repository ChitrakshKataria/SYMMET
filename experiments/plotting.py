import matplotlib.pyplot as plt





### Loss graph
def LossGraph(avg_loss_train, avg_loss_test):
    """
    COMMING SOON
    """
    plt.figure()

    plt.plot(avg_loss_train, color="royalblue", linewidth=2, label="Training Loss")
    plt.plot(avg_loss_test, color="red", linewidth=2, label="Testing Loss")

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Neural Network Loss")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


### Acc graph
def AccuracyGraph(avg_train_accuracy, avg_test_accuracy):
    """
    COMMING SOON
    """
    plt.figure()

    plt.plot(avg_train_accuracy, color="royalblue", linewidth=2, label="Training Acc")
    plt.plot(avg_test_accuracy, color="red", linewidth=2, label="Testing Acc")

    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy")
    plt.title("Neural Network Accuracy")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
