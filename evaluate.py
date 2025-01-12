# Import necessary libraries
import torch  # Main PyTorch library
from tqdm import tqdm  # For progress bars
import numpy as np  # For numerical operations
from utils import load_checkpoint  # Custom function to load saved models

def evaluate_model(model, test_loader, device, criterion=None):
    """
    This function tests how well our trained model performs on new data.
    
    Parameters:
    - model: Our trained neural network
    - test_loader: Provides batches of test images
    - device: Whether to use CPU or GPU ('cuda' or 'cpu')
    - criterion: The loss function (optional)
    """
    # Set model to evaluation mode - this disables dropout and batch normalization
    # These are techniques used during training but not needed during testing
    model.eval()
    
    # Initialize counters for overall accuracy
    correct = 0  # Number of correct predictions
    total = 0    # Total number of predictions
    running_loss = 0.0  # Accumulator for loss values
    
    # Initialize counters for per-class accuracy
    # We have 10 classes (plane, car, bird, etc.), so make a list of 10 zeros
    class_correct = list(0. for i in range(10))  # Correct predictions per class
    class_total = list(0. for i in range(10))    # Total predictions per class
    
    # torch.no_grad() tells PyTorch not to track gradients
    # We don't need gradients for testing - this saves memory and speeds up testing
    with torch.no_grad():
        # Loop through batches of test images
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            # Move images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)
            
            # Get model predictions
            # outputs shape: [batch_size, num_classes]
            # Each row contains scores for each class
            outputs = model(images)
            
            # Calculate loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()  # Add batch loss to total
            
            # Get predicted class for each image
            # torch.max returns (values, indices) - we only want indices
            # _ means we're ignoring the values
            _, predicted = torch.max(outputs, 1)  # 1 means along row dimension
            
            # Update overall accuracy counters
            total += labels.size(0)  # Add batch size to total
            correct += (predicted == labels).sum().item()  # Add correct predictions
            
            # Calculate per-class accuracy
            c = (predicted == labels).squeeze()  # Get boolean array of correct predictions
            for i in range(len(labels)):
                label = labels[i]  # Get true class
                class_correct[label] += c[i].item()  # Add to correct if prediction was right
                class_total[label] += 1  # Increment total for this class
    
    # Calculate final metrics
    accuracy = 100 * correct / total  # Convert to percentage
    # Calculate average loss if criterion was provided
    avg_loss = running_loss / len(test_loader) if criterion is not None else None
    
    # Print overall results
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    if avg_loss is not None:
        print(f"Average Test Loss: {avg_loss:.4f}")
    
    # Define class names for CIFAR-10 dataset
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Print accuracy for each class
    print("\nPer-class Accuracy:")
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]  # Calculate percentage
        print(f'{classes[i]:>10s}: {class_acc:.2f}%')  # :>10s means right-align with width 10
    
    return accuracy, avg_loss

def load_and_evaluate(model, checkpoint_path, test_loader, device, criterion=None):
    """
    Helper function that loads a saved model and evaluates it.
    
    Parameters:
    - model: The initial model architecture (not trained)
    - checkpoint_path: Path to the saved model file
    - test_loader: Provides batches of test images
    - device: Whether to use CPU or GPU
    - criterion: Loss function (optional)
    """
    # Load the saved model weights and metadata
    model, epoch, val_loss = load_checkpoint(model, checkpoint_path)
    
    # Move model to GPU if available
    model = model.to(device)
    
    # Print info about the loaded checkpoint
    print(f"\nLoaded checkpoint from epoch {epoch} with validation loss {val_loss:.4f}")
    
    # Evaluate the loaded model
    return evaluate_model(model, test_loader, device, criterion)