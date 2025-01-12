# Import required PyTorch libraries
import torch  # Main PyTorch library
from torch.utils.data import DataLoader  # Handles batching of data
from torchvision import datasets, transforms  # Contains common datasets and image transformations

def get_cifar10_dataloader(batch_size=128):
    """
    Creates data loaders for the CIFAR-10 dataset.
    CIFAR-10 contains 60,000 32x32 color images in 10 different classes:
    [airplane, car, bird, cat, deer, dog, frog, horse, ship, truck]
    
    Args:
        batch_size: Number of images to process at once (default: 128)
    """
    
    # Define the transformations to apply to training images
    # Transformations help prevent overfitting by creating variations of the training images
    train_transform = transforms.Compose([  # Compose combines multiple transforms
        # Randomly crop a 32x32 patch from a 40x40 padded image (padding=4)
        # This helps the model learn to recognize objects in different positions
        transforms.RandomCrop(32, padding=4),
        
        # Randomly flip images horizontally with 0.5 probability
        # This helps the model learn that left/right orientation doesn't matter
        transforms.RandomHorizontalFlip(),
        
        # Randomly adjust image colors
        # This helps the model learn to handle different lighting conditions
        transforms.ColorJitter(
            brightness=0.2,  # Adjust brightness by ±20%
            contrast=0.2,    # Adjust contrast by ±20%
            saturation=0.2   # Adjust saturation by ±20%
        ),
        
        # Convert PIL images to PyTorch tensors (values between 0 and 1)
        transforms.ToTensor(),
        
        # Normalize the images using CIFAR-10's mean and standard deviation
        # This standardizes the input to help training converge faster
        # Format: ((mean_R, mean_G, mean_B), (std_R, std_G, std_B))
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),  # Mean values for each RGB channel
            (0.2023, 0.1994, 0.2010)   # Standard deviation for each RGB channel
        )
    ])
    
    # Define transformations for test images
    # We don't use augmentation for test data because we want consistent results
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        # Use same normalization as training for consistency
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # Load the training dataset
    train_dataset = datasets.CIFAR10(
        root='./data',           # Directory to store/load the dataset
        train=True,              # Get the training split
        download=True,           # Download if not already present
        transform=train_transform # Apply our training transformations
    )
    
    # Load the test dataset
    test_dataset = datasets.CIFAR10(
        root='./data',          # Same directory as training data
        train=False,            # Get the test split
        download=True,          # Download if not already present
        transform=test_transform # Apply our test transformations
    )

    # Create the training data loader
    train_loader = DataLoader(
        train_dataset,          # Our training dataset
        batch_size=batch_size,  # Number of images per batch
        shuffle=True,           # Shuffle the data each epoch
        num_workers=4,          # Number of parallel processes for data loading
        pin_memory=True         # Speeds up data transfer to GPU if using CUDA
    )
    
    # Create the test data loader
    test_loader = DataLoader(
        test_dataset,           # Our test dataset
        batch_size=batch_size,  # Same batch size as training
        shuffle=False,          # Don't shuffle test data
        num_workers=4,          # Same number of workers as training
        pin_memory=True         # Same memory setting as training
    )

    # Return both data loaders and the class names
    # train_loader: yields batches of training data
    # test_loader: yields batches of test data
    # train_dataset.classes: list of class names ['airplane', 'car', etc.]
    return train_loader, test_loader, train_dataset.classes


"""
# Example usage:
# train_loader, test_loader, classes = get_cifar10_dataloader(batch_size=128)

# Each batch from train_loader or test_loader will contain:
# - images tensor of shape (batch_size, 3, 32, 32) where:
#   - batch_size is number of images (e.g., 128)
#   - 3 is number of color channels (RGB)
#   - 32, 32 is the image dimensions
# - labels tensor of shape (batch_size,) containing class indices (0-9)
"""