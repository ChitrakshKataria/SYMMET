from torch import nn
import torch

def train_loop(model,
               X_train,
               y_train,
               X_test,
               y_test,
               device,
               epochs=100,
               lr=0.001,
               batch_size=128):
    """
    Train and Evaluate a PyTorch neural network using mini-batch gradient descent.

    Uses the Adam optimizer and BCEWithLogitsLoss for binary classification.
    Records training and testing loss and accuracy after eatch epoch.

    Args:
        model (nn.Module): PyTorch model to train and evaluate.
        X_train (torch.Tensor): Training input fetures.
        y_train (torch.Tensor): Training labels (0 or 1).
        X_test (torch.Tensor): Testing input fetures.
        y_test (torch.Tensor): Testing labels (0 or 1).
        device (Object): Device model will be running on
        epochs (int): Number of iterations on training data. Defaults to 100.
        lr (float): Learning Rate for the Adam optimizer. Defaults to 0.001.
        batch_size (int): Number of samples per mini-batch. Defaults to 128.
    Returns:
        dict: Dictionary containing:
            model (nn.Module): The trained PyTorch model.
            train_loss (list[float]): Average training loss per epoch.
            test_loss (loss[float]): Average testing loss per epoch.
            train_accuracy (list[float]): Training accuracy (%) per epoch.
            test_accuracy (list[flot]): Testing accuracy (%) per epoch.
    """
    
    ### Transfering the training and testing data to the GPU if required
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    X_test= X_test.to(device)
    y_test = y_test.to(device)

    ### Init Model
    running_model  = model.to(device)


    ### Hyperparamaters loop
    epochs = epochs
    lr = lr
    batch = batch_size


    ### Init Optim and loss
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(running_model.parameters(), lr=lr)


    ### Empty Variables 
    avg_loss_train = []
    avg_loss_test = []
    avg_test_accuracy = []
    avg_train_accuracy = []



    ### Training/Testing loop
    for epoch in range(epochs):
        running_model.train()

        loss_train = []
        loss_test = []
        correct_predictions = 0

        ### Shuffle the training data (Mini-Batch)
        indecies = torch.randperm(len(X_train))
        X_train = X_train[indecies]
        y_train = y_train[indecies]

        ## Training Loop (Mini-Batch)
        for start in range(0, len(X_train), batch):
            end = start + batch
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            train_logits = running_model(X_batch)

            loss = loss_fn(train_logits, y_batch.unsqueeze(dim=1))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            
            ### Accuracy
            predicted_class = torch.sigmoid(train_logits).round()
            correct_predictions += (predicted_class == y_batch.unsqueeze(1)).sum().item()

            ### Store training loss
            loss_train.append(loss.item())

        ### Average training accuracy and loss (per epoch)
        train_accuracy = correct_predictions / len(X_train) * 100
        avg_train_accuracy.append(train_accuracy)

        train_loss = sum(loss_train) / len(loss_train)
        avg_loss_train.append(train_loss)

        ### Evaluation
        running_model.eval()

        with torch.no_grad():
            test_logits = running_model(X_test)
            test_preds = torch.sigmoid(test_logits).round()

            ### Test accuracy
            test_accuracy = (test_preds == y_test.unsqueeze(1)).float().mean() * 100

            avg_test_accuracy.append(test_accuracy.item())

            ### Test loss
            test_loss = loss_fn(test_logits, y_test.unsqueeze(1))
            avg_loss_test.append(test_loss.item())

    return{
        "model": running_model,
        "train_loss": avg_loss_train,
        "test_loss": avg_loss_test,
        "train_accuracy": avg_train_accuracy,
        "test_accuracy": avg_test_accuracy
    }