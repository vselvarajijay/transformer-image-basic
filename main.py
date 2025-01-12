import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime

from model import VisionTransformer
from dataset import get_cifar10_dataloader
from train import train_model
from evaluate import load_and_evaluate
from utils import plot_training_history, get_latest_checkpoint

def setup_parser():
    parser = argparse.ArgumentParser(description='Vision Transformer for CIFAR-10')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help='Mode: train a new model or test an existing one')
    
    # Model parameters
    parser.add_argument('--embed-dim', type=int, default=192,
                        help='Embedding dimension (default: 192)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--num-layers', type=int, default=12,
                        help='Number of transformer layers (default: 12)')
    parser.add_argument('--mlp-ratio', type=int, default=4,
                        help='MLP expansion ratio (default: 4)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience (default: 7)')
    
    # Paths
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints (default: checkpoints)')
    parser.add_argument('--checkpoint-path', type=str,
                        help='Path to specific checkpoint for testing (optional)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    return parser

def train(args):
    print("\n=== Starting Training ===\n")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader, classes = get_cifar10_dataloader(batch_size=args.batch_size)
    
    # Initialize model
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.dropout,
        attn_drop_rate=args.dropout
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.weight_decay)
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Plot training history
    plot_training_history(history)
    print("\n=== Training Completed ===\n")

def test(args):
    print("\n=== Starting Evaluation ===\n")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    _, test_loader, classes = get_cifar10_dataloader(batch_size=args.batch_size)
    
    # Initialize model architecture
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.dropout,
        attn_drop_rate=args.dropout
    )
    
    # Get checkpoint path
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path is None:
            print("No checkpoint found! Please provide a checkpoint path or train a model first.")
            return
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    accuracy, loss = load_and_evaluate(model, checkpoint_path, test_loader, device, criterion)
    
    print("\n=== Evaluation Completed ===\n")

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        test(args)

if __name__ == "__main__":
    main()    