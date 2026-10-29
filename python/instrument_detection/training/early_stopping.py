"""
Early stopping for the training loop. Saves a checkpoint whenever validation loss improves
and sets a flag to stop once it has not improved for a number of epochs.
"""
import torch
import os

class EarlyStopping:
    """
    Implements early stopping to halt training when validation loss stops decreasing.

    Attributes:
        patience (int): Number of epochs to wait before stopping if no improvement.
        delta (float): Minimum change in validation loss to qualify as improvement.
        save_path (str): Path to save the best model checkpoint.
        best_loss (float or None): Best validation loss observed.
        counter (int): Number of consecutive epochs without improvement.
        early_stop (bool): Flag indicating whether training should be stopped.
        metric_name (str): Name of the metric being monitored.
    """

    def __init__(self, patience=5, delta=0.001, save_path='', metric_name='Loss'):
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int): Number of epochs to wait before stopping.
            delta (float): Minimum improvement required to reset patience.
            save_path (str): File path to save the best model.
            metric_name (str): Name of the metric for logging purposes.
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.metric_name = metric_name
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, optimizer, scaler, epoch, thresholds=None):
        """
        Checks validation loss and determines whether to stop training.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
            epoch (int): The current training epoch.
        """
        loss = val_loss

        # If no best loss has been recorded, initialize it and save the checkpoint.
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model, optimizer, scaler, epoch, thresholds)

        # If the loss does not decrease beyond delta, increase the counter.
        elif loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            print(f"Current {self.metric_name}: {loss:.6f}, Best {self.metric_name}: {self.best_loss:.6f}")

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # If improvement is found, update best loss and reset counter
            improvement = self.best_loss - loss
            self.best_loss = loss
            self.save_checkpoint(model, optimizer, scaler, epoch, thresholds)
            self.counter = 0
            print(f"{self.metric_name} improved by {improvement:.6f} to {loss:.6f}")

    def save_checkpoint(self, model, optimizer, scaler, epoch, thresholds=None):
        """
        Saves the current model checkpoint.

        Args:
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer state to save.
            scaler (torch.cuda.amp.GradScaler): The scaler state for mixed precision training.
            epoch (int): The epoch at which the checkpoint is saved.
            thresholds (np.ndarray, optional): The thresholds used for evaluation.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': self.best_loss,
            'metric_name': self.metric_name
        }

        # Only save scaler if it's provided
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        # Save thresholds if provided
        if thresholds is not None:
            checkpoint['thresholds'] = thresholds.tolist() if hasattr(thresholds, 'tolist') else thresholds

        torch.save(checkpoint, self.save_path)
        print(f"Validation {self.metric_name} improved. Model saved to {self.save_path}")

    def load_checkpoint(self, model, optimizer=None, scaler=None):
        """
        Load a saved checkpoint.

        Args:
            model (torch.nn.Module): The model to load state into.
            optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
            scaler (torch.cuda.amp.GradScaler, optional): The scaler to load state into.

        Returns:
            dict: Checkpoint information including epoch, best loss, and thresholds.
        """
        if not os.path.exists(self.save_path):
            print(f"No checkpoint found at {self.save_path}")
            return None

        checkpoint = torch.load(self.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_loss = checkpoint.get('best_loss', None)
        thresholds = checkpoint.get('thresholds', None)

        print(f"Checkpoint loaded from {self.save_path}")
        print(f"Epoch: {checkpoint['epoch']}, Best {self.metric_name}: {self.best_loss:.6f}")

        if thresholds is not None:
            print(f"Loaded thresholds: {thresholds}")

        return checkpoint