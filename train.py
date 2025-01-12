# Import required libraries
import torch  # PyTorch deep learning library
from tqdm import tqdm  # For progress bars during training
import os  # For file and directory operations
from datetime import datetime  # For timestamps in saved files

class EarlyStopping:
    """
    Early Stopping checks if the model's performance has stopped improving and stops training if it has.
    This prevents overfitting, which is when a model performs well on training data but poorly on new data.
    
    Think of it like knowing when to stop studying - if your practice test scores (validation loss)
    haven't improved in a while, studying more (training more) probably won't help.
    """
    def __init__(self, patience=7, min_delta=0, checkpoint_path='checkpoints'):
        # patience: number of epochs to wait before stopping
        # e.g., if patience=7, we'll wait 7 epochs of no improvement before stopping
        self.patience = patience
        
        # min_delta: minimum change that counts as an improvement
        # e.g., if min_delta=0.1, a change from 1.0 to 0.95 isn't considered an improvement
        self.min_delta = min_delta
        
        # counter: keeps track of how many epochs we've gone without improvement
        self.counter = 0
        
        # best_loss: keeps track of the best validation loss we've seen
        # None at start because we haven't seen any losses yet
        self.best_loss = None
        
        # early_stop: flag that tells us when to stop training
        self.early_stop = False
        
        # checkpoint_path: where to save model checkpoints
        self.checkpoint_path = checkpoint_path
        
        # Create the checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_path, exist_ok=True)
        
    def __call__(self, val_loss, model, epoch, optimizer):
        """
        Called at the end of each epoch to check if we should stop training.
        
        Args:
            val_loss: The current validation loss (how badly the model is performing)
            model: The neural network we're training
            epoch: The current training epoch number
            optimizer: The optimizer that's updating the model's weights
        """
        # First epoch is a special case - everything is the "best so far"
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, epoch, val_loss, optimizer)
        
        # If the current loss is worse than the best loss (plus some minimum improvement)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1  # Add to our "epochs without improvement" counter
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            # If we've been patient long enough, stop training
            if self.counter >= self.patience:
                self.early_stop = True
        
        # If the current loss is better than the best loss
        else:
            self.best_loss = val_loss  # Update our "best loss" record
            self.save_checkpoint(model, epoch, val_loss, optimizer)  # Save this good model
            self.counter = 0  # Reset our patience counter
            
    def save_checkpoint(self, model, epoch, val_loss, optimizer):
        """
        Saves the model's current state when it performs better than ever before.
        Like taking a snapshot of the model at its best moment.
        
        The checkpoint contains everything needed to resume training:
        - The model's learned parameters
        - The optimizer's state
        - The epoch number
        - The validation loss at this point
        """
        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.checkpoint_path, f'model_epoch{epoch}_{timestamp}.pt')
        
        # Save all the important information
        torch.save({
            'epoch': epoch,  # Which epoch this was
            'model_state_dict': model.state_dict(),  # The model's learned parameters
            'optimizer_state_dict': optimizer.state_dict(),  # The optimizer's state
            'val_loss': val_loss,  # How well the model was performing
        }, filepath)
        
        print(f'Checkpoint saved: {filepath}')

def train_model(model, train_loader, test_loader, optimizer, criterion, device, 
                num_epochs=100, patience=7, checkpoint_dir='checkpoints'):
    """
    The main training function that teaches our model to recognize images.
    
    Think of this as a teacher (optimizer) helping a student (model) learn from a textbook
    (train_loader) and taking tests (test_loader) to check their progress.
    
    Args:
        model: The neural network we're training (our student)
        train_loader: Provides training data (our textbook)
        test_loader: Provides validation data (our tests)
        optimizer: Updates model weights (our teaching method)
        criterion: Measures how wrong the model is (our grading system)
        device: Whether to use CPU or GPU (our study environment)
        num_epochs: How many times to go through the training data (study sessions)
        patience: How long to wait before early stopping (when to take a break)
        checkpoint_dir: Where to save model checkpoints (progress reports)
    """
    # Initialize early stopping to prevent overfitting
    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_dir)
    
    # Setup learning rate scheduler
    # This gradually decreases how big the model's adjustments are, like gradually
    # making a student solve harder and more precise problems
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Dictionary to store all our training history
    # This is like keeping a gradebook of all scores and progress
    history = {
        'train_loss': [], 'train_acc': [],  # Training performance
        'val_loss': [], 'val_acc': [],      # Validation performance
        'lr': []                            # Learning rate changes
    }
    
    # Main training loop - each epoch is one complete pass through the dataset
    for epoch in range(num_epochs):
        # Set model to training mode
        # This is like telling the student "it's study time"
        model.train()
        
        # Initialize tracking variables for this epoch
        running_loss = 0.0  # Accumulator for total loss
        correct = 0  # Counter for correct predictions
        total = 0    # Counter for total predictions
        
        # Training Loop - This is where the actual learning happens
        # tqdm gives us a nice progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for images, labels in pbar:
            # Move data to the right device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass: Get model predictions
            # This is like the student trying to answer questions
            outputs = model(images)
            
            # Calculate loss (how wrong the model was)
            # This is like grading the student's answers
            loss = criterion(outputs, labels)
            
            # Reset gradients from previous batch
            # This is like clearing the student's scratch paper
            optimizer.zero_grad()
            
            # Backward pass: Calculate gradients
            # This is like figuring out what the student did wrong
            loss.backward()
            
            # Clip gradients to prevent them from exploding
            # This keeps the learning stable, like preventing the student from
            # making too drastic changes to their approach
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update model weights
            # This is like the student learning from their mistakes
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()  # Add this batch's loss
            _, predicted = outputs.max(1)  # Get predicted classes
            total += labels.size(0)  # Count total predictions
            correct += predicted.eq(labels).sum().item()  # Count correct predictions
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss/(pbar.n+1),
                'acc': 100.*correct/total
            })
        
        # Calculate average training metrics
        train_loss = running_loss/len(train_loader)
        train_acc = 100.*correct/total
        
        # Validation Loop - Test how well the model is learning
        # This is like giving the student a practice test
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        # We don't need gradients for validation
        # This is like telling the student "this is just a practice test"
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
        
        # Store metrics in history
        # This is like writing scores in our gradebook
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        
        # Update learning rate
        # This is like adjusting how much help we give the student
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Check if we should stop training
        # This is like deciding if more studying would help
        early_stopping(val_loss, model, epoch, optimizer)
        if early_stopping.early_stop:
            print("Early stopping triggered - validation loss stopped improving")
            break
            
    return history  # Return all the training history for analysis