# Standard PyTorch imports
import torch
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Activation functions and other utilities

class LayerNorm(nn.Module):
    """
    Layer Normalization helps stabilize training by normalizing the inputs across features.
    Think of it like standardizing data, but for deep learning layers.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps  # Small number for numerical stability
        # Learnable parameters for scaling (gamma) and shifting (beta)
        self.gamma = nn.Parameter(torch.ones(dim))   # Multiplicative
        self.beta = nn.Parameter(torch.zeros(dim))   # Additive

    def forward(self, x):
        # Calculate mean and variance along last dimension
        mean = x.mean(-1, keepdim=True)     # Average of features
        var = x.var(-1, keepdim=True, unbiased=False)  # Variance of features
        # Normalize and scale/shift
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.
    Processes images by:
    1. Splitting them into patches
    2. Converting patches to embeddings
    3. Adding positional information
    4. Processing through transformer layers
    5. Making classification prediction
    """
    def __init__(self, image_size=32, patch_size=4, num_classes=10, 
                 embed_dim=192, depth=12, num_heads=8, mlp_ratio=4., 
                 drop_rate=0.1, attn_drop_rate=0.1):
        """
        Parameters:
        - image_size: Input image size (32 for CIFAR-10)
        - patch_size: Size of image patches (4x4 pixels)
        - num_classes: Number of output classes (10 for CIFAR-10)
        - embed_dim: Dimension of token embeddings
        - depth: Number of transformer layers
        - num_heads: Number of attention heads
        - mlp_ratio: Expansion ratio for MLP layer
        - drop_rate: Dropout rate
        - attn_drop_rate: Attention dropout rate
        """
        super().__init__()
        
        # Calculate number of patches (for 32x32 image and 4x4 patches = 64 patches)
        num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding: Convert image patches to embeddings using convolution
        # Input: (batch_size, 3, 32, 32)
        # Output: (batch_size, 64, embed_dim)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable classification token (added to start of each sequence)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learnable position embeddings (added to patch embeddings)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Dropout layer for regularization
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Create transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,        # Dimension of embeddings
            nhead=num_heads,          # Number of attention heads
            dim_feedforward=int(embed_dim * mlp_ratio),  # MLP hidden dimension
            dropout=drop_rate,        # Dropout rate
            activation='gelu',        # Activation function
            batch_first=True,         # Batch dimension first
            norm_first=True           # Apply normalization before attention
        )
        
        # Stack multiple transformer layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Final classification head
        self.norm = LayerNorm(embed_dim)  # Final normalization
        self.pre_logits = nn.Linear(embed_dim, embed_dim)  # Additional linear layer
        self.head = nn.Linear(embed_dim, num_classes)  # Classification layer
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)   # Position embeddings
        nn.init.trunc_normal_(self.cls_token, std=0.02)   # Classification token
        self.apply(self._init_weights)  # Initialize all other weights
        
    def _init_weights(self, m):
        """Initialize weights for linear and normalization layers"""
        if isinstance(m, nn.Linear):
            # Initialize linear layer weights with truncated normal distribution
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # Initialize biases to zero
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)    # Initialize LayerNorm bias to zero
            nn.init.ones_(m.weight)   # Initialize LayerNorm weight to one
            
    def forward(self, x):
        """
        Forward pass of the model
        Input: Image tensor of shape (batch_size, 3, image_size, image_size)
        Output: Class logits of shape (batch_size, num_classes)
        """
        B = x.shape[0]  # Batch size
        
        # 1. Convert image to patches and embed
        # Input: (B, 3, 32, 32) -> Output: (B, num_patches, embed_dim)
        x = self.patch_embed(x)  # Convert patches to embeddings
        x = x.flatten(2).transpose(1, 2)  # Reshape to sequence
        
        # 2. Add classification token and position embeddings
        cls_token = self.cls_token.expand(B, -1, -1)  # Expand cls token to batch size
        x = torch.cat((cls_token, x), dim=1)  # Add cls token to start
        x = x + self.pos_embed  # Add position information
        x = self.pos_drop(x)    # Apply dropout
        
        # 3. Process through transformer layers
        x = self.transformer(x)
        
        # 4. Classification head
        x = self.norm(x[:, 0])      # Take cls token output and normalize
        x = self.pre_logits(x)      # Additional linear layer
        x = F.gelu(x)               # Non-linear activation
        x = self.head(x)            # Final classification
        
        return x  # Return class logits