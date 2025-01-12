import torch
from tqdm import tqdm  # Progress bar library
import os
from datetime import datetime

class EarlyStopping:
    """
    Stops training when the model stops improving on validation data.
    This helps prevent overfitting (when model performs well on training data but poorly on new data).
    """
    def __init__(self, patience=7, min_delta=0, checkpoint_path='checkpoints'):
        # patience: number of epochs to wait before stopping when model isn't improving
        self.patience = patience
        # min_delta: minimum change in validation loss to be considered an improvement
        self.min_delta = min_delta
        # counter: keeps track of epochs without improvement
        self.counter = 0
        # best_loss: stores the best validation loss seen so far
        self.best_loss = None
        # early_stop: flag to indicate if training should stop
        self.early_stop = False
        # checkpoint_path: directory to save model checkpoints
        self.checkpoint_path = checkpoint_path
        # create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_path, exist_ok=True)
        
    def __call__(self, val_loss, model, epoch, optimizer):
        """Called after each epoch to check if training should stop"""
        # First epoch, initialize best loss
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, epoch, val_loss, optimizer)
        # If validation loss got worse
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1  # increment patience counter
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  # stop training if patience exceeded
        # If validation loss improved
        else:
            self.best_loss = val_loss  # update best loss
            self.save_checkpoint(model, epoch, val_loss, optimizer)
            self.counter = 0  # reset patience counter
            
    def save_checkpoint(self, model, epoch, val_loss, optimizer):
        """Saves model state when validation loss improves"""
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.checkpoint_path, f'model_epoch{epoch}_{timestamp}.pt')
        # Save model state, optimizer state, and metadata
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),  # model's learned parameters
            'optimizer_state_dict': optimizer.state_dict(),  # optimizer's state
            'val_loss': val_loss,
        }, filepath)
        print(f'Checkpoint saved: {filepath}')

def train_model(model, train_loader, test_loader, optimizer, criterion, device, 
                num_epochs=100, patience=7, checkpoint_dir='checkpoints'):
    """
    Main training function that handles the entire training process.
    
    Parameters:
        model: Neural network model to train
        train_loader: Provides batches of training data
        test_loader: Provides batches of validation data
        optimizer: Updates model weights (e.g., Adam)
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Computing device ('cuda' for GPU, 'cpu' for CPU)
        num_epochs: Number of complete passes through the training data
        patience: How many epochs to wait before early stopping
        checkpoint_dir: Where to save model checkpoints
    """
    # Initialize early stopping to prevent overfitting
    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_dir)
    
    # Setup learning rate scheduler that gradually decreases learning rate
    # This helps fine-tune the model as training progresses
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Dictionary to store training metrics for later plotting
    history = {
        'train_loss': [], 'train_acc': [],  # training metrics
        'val_loss': [], 'val_acc': [],      # validation metrics
        'lr': []                            # learning rates
    }
    
    # Main training loop - each epoch is one complete pass through the dataset
    for epoch in range(num_epochs):
        # Set model to training mode (enables dropout, batch norm, etc.)
        model.train()
        running_loss = 0.0  # accumulator for loss in current epoch
        correct = 0  # counter for correct predictions
        total = 0    # counter for total predictions
        
        # Training Loop - process training data in batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for images, labels in pbar:
            # Move data to appropriate device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass: feed images through model to get predictions
            outputs = model(images)  # outputs shape: [batch_size, num_classes]
            
            # Calculate loss between model predictions and true labels
            loss = criterion(outputs, labels)
            
            # Reset gradients from previous batch
            optimizer.zero_grad()
            
            # Backward pass: calculate gradients of the loss
            loss.backward()
            
            # Clip gradients to prevent them from exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update model weights using the optimizer
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()  # accumulate batch loss
            _, predicted = outputs.max(1)  # get predicted class (highest probability)
            total += labels.size(0)  # count total predictions
            correct += predicted.eq(labels).sum().item()  # count correct predictions
            
            # Update progress bar with current statistics
            pbar.set_postfix({
                'loss': running_loss/(pbar.n+1),
                'acc': 100.*correct/total
            })
        
        # Calculate average training metrics for this epoch
        train_loss = running_loss/len(train_loader)
        train_acc = 100.*correct/total
        
        # Validation Loop - evaluate model on validation data
        model.eval()  # set model to evaluation mode (disables dropout, etc.)
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        # Process validation data without computing gradients
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate average validation metrics
        val_loss = val_loss/len(test_loader)
        val_acc = 100.*val_correct/val_total
        
        # Store metrics in history for plotting
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        
        # Update learning rate using scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Check if we should stop training (early stopping)
        early_stopping(val_loss, model, epoch, optimizer)
        if early_stopping.early_stop:
            print("Early stopping triggered - validation loss stopped improving")
            break
            
    return history  # return metrics for plotting