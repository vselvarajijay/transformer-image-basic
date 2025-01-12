import torch
from tqdm import tqdm
import numpy as np
from utils import load_checkpoint

def evaluate_model(model, test_loader, device, criterion=None):
    """
    Evaluates the model on the test dataset and returns accuracy and loss.
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader) if criterion is not None else None
    
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    if avg_loss is not None:
        print(f"Average Test Loss: {avg_loss:.4f}")
    
    # Print per-class accuracy
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    print("\nPer-class Accuracy:")
    for i in range(10):
        print(f'{classes[i]:>10s}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    return accuracy, avg_loss

def load_and_evaluate(model, checkpoint_path, test_loader, device, criterion=None):
    """
    Loads a checkpoint and evaluates the model
    """
    model, epoch, val_loss = load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    print(f"\nLoaded checkpoint from epoch {epoch} with validation loss {val_loss:.4f}")
    
    return evaluate_model(model, test_loader, device, criterion)