# Import necessary libraries
# torch: The main deep learning library we're using
# transforms: Tools for processing images (resize, normalize, etc.)
# PIL (Python Imaging Library): For loading and working with images
# os: For working with files and directories
import torch
from torchvision import transforms
from PIL import Image
import os
from model import VisionTransformer

def load_image(image_path):
    """
    Prepare an image for the AI model to analyze.
    
    This function does three main things:
    1. Resizes any image to 32x32 pixels (what our model expects)
    2. Converts the image into a format the model can understand (tensor)
    3. Normalizes the image colors (helps the model analyze better)
    
    Think of this like preparing a dish the same way every time
    so the food critic (our model) can judge consistently.
    """
    # Create a set of image transformations that will run in sequence
    transform = transforms.Compose([
        # Resize the image to 32x32 pixels (what CIFAR-10 model expects)
        transforms.Resize((32, 32)),
        
        # Convert image to PyTorch tensor (a format the model understands)
        # This also converts pixel values from 0-255 to 0-1
        transforms.ToTensor(),
        
        # Normalize the colors using CIFAR-10 dataset's mean and std values
        # This helps the model process images more effectively
        transforms.Normalize(
            # Mean values for each color channel (R,G,B)
            mean=[0.4914, 0.4822, 0.4465],
            # Standard deviation for each color channel
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # Open the image and convert to RGB (some images might be grayscale or RGBA)
    image = Image.open(image_path).convert('RGB')
    
    # Apply our transformations and add a batch dimension
    # The batch dimension is needed because models expect to process multiple images at once
    # even when we only have one image
    return transform(image).unsqueeze(0)

def predict_image(model, image_tensor, device):
    """
    Use the AI model to predict what's in the image.
    
    Parameters:
        model: The trained AI model
        image_tensor: The processed image
        device: Whether to use CPU or GPU for prediction
    
    Returns:
        A number representing which category the model thinks the image belongs to
    """
    # Set model to evaluation mode (like telling the AI "time for a test, not practice")
    model.eval()
    
    # Tell PyTorch not to calculate gradients (we don't need them for predictions)
    # This saves memory and makes predictions faster
    with torch.no_grad():
        # Move the image to the same device (CPU/GPU) as the model
        image_tensor = image_tensor.to(device)
        
        # Get the model's prediction
        outputs = model(image_tensor)
        
        # The model outputs probabilities for each category
        # We take the category with highest probability as our answer
        _, predicted = torch.max(outputs, 1)
        
        # Return just the category number
        return predicted.item()

def get_latest_checkpoint(checkpoint_dir):
    """
    Find the most recently saved model in the checkpoints directory.
    
    A checkpoint is like a saved state in a video game - it contains
    all the knowledge the model has learned up to that point.
    """
    # Check if the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Get all files ending with .pt (PyTorch model files)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        return None
    
    # Sort files and get the latest one
    # Our filenames contain timestamps, so sorting gives us the most recent
    latest_checkpoint = sorted(checkpoint_files)[-1]
    return os.path.join(checkpoint_dir, latest_checkpoint)

def setup_inference(model, checkpoint_path, device='cuda'):
    """
    Prepare the model for making predictions.
    
    This function:
    1. Defines what categories the model knows about
    2. Loads the model's learned knowledge from a checkpoint file
    3. Prepares the model for making predictions
    """
    # List of categories our model can recognize
    # These are the classes from the CIFAR-10 dataset the model was trained on
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load the saved model state
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Update the model with learned weights from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to the appropriate device (GPU if available, else CPU)
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, classes

def main():
    """
    The main function that runs the whole process:
    1. Sets up the AI model
    2. Finds images to analyze
    3. Makes predictions for each image
    """
    # Determine if we can use a GPU (cuda) or need to use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "checkpoints"  # Where to find saved model states
    
    # Find the most recent saved model state
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        print("No checkpoint files found in directory:", checkpoint_dir)
        return
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Create a new Vision Transformer model
    # These parameters define the model's architecture (like its brain structure)
    model = VisionTransformer(
        image_size=32,      # Size of input images
        patch_size=4,       # Size of image patches the model looks at
        num_classes=10,     # Number of categories it can recognize
        embed_dim=192,      # Size of internal representations
        depth=12,           # Number of transformer layers
        num_heads=8,        # Number of attention heads
        mlp_ratio=4,        # Size of internal neural networks
        drop_rate=0.1,      # Dropout rate for regularization
        attn_drop_rate=0.1  # Attention dropout rate
    )
    
    # Load the trained model state
    model, classes = setup_inference(model, checkpoint_path, device)
    
    # Set up directory for images we want to classify
    image_dir = "custom_images"
    
    # Create the image directory if it doesn't exist
    if not os.path.exists(image_dir):
        print(f"Creating directory for custom images: {image_dir}")
        os.makedirs(image_dir)
        print(f"Please put your images in the {image_dir} directory and run again.")
        return
    
    # Find all image files in the directory
    # We only look for common image formats: PNG, JPG, JPEG
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Check if we found any images
    if not image_files:
        print(f"No image files found in {image_dir}")
        print("Please add some images (PNG, JPG, or JPEG) to the directory and run again.")
        return
    
    # Process each image we found
    print("\nProcessing images...")
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # Load and prepare the image
            image_tensor = load_image(image_path)
            
            # Get the model's prediction
            pred_idx = predict_image(model, image_tensor, device)
            predicted_class = classes[pred_idx]
            
            # Print the result
            print(f"\nImage: {image_file}")
            print(f"Prediction: {predicted_class}")
            
        except Exception as e:
            # If anything goes wrong with this image, print the error
            print(f"\nError processing {image_file}: {str(e)}")

# This is where the program starts running
if __name__ == "__main__":
    main()