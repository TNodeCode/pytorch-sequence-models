import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding import EmbeddingType
from models.classifier import Classifier


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (SSM) block - the core component of Mamba.
    This implements a simplified version of the selective SSM mechanism.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        self.d_conv = d_conv
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolutional layer for temporal mixing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + self.d_inner, bias=False)
        
        # SSM state initialization
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and split
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_inner, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # Causal convolution
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # SSM parameters from input
        ssm_params = self.x_proj(x_conv)  # (B, L, d_state + d_state + d_inner)
        delta, B, C = ssm_params.split([self.d_state, self.d_state, self.d_inner], dim=-1)
        
        # Discretization
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # Selective scan (simplified version)
        # In full Mamba, this would be a parallel scan
        y = self._selective_scan(x_conv, delta, B, C)
        
        # Gating
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def _selective_scan(self, x, delta, B, C):
        """
        Simplified selective scan operation.
        In practice, this would use parallel scan algorithms for efficiency.
        """
        batch_size, seq_len, d_inner = x.shape
        
        # Initialize state
        h = torch.zeros(batch_size, self.d_state, d_inner, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # Current input
            x_t = x[:, t, :]  # (B, d_inner)
            delta_t = delta[:, t, :]  # (B, d_inner)
            B_t = B[:, t, :]  # (B, d_state)
            C_t = C[:, t, :]  # (B, d_inner)
            
            # Update state: h_t = delta_t * h_{t-1} + B_t * x_t
            # Simplified: we use broadcasting and element-wise operations
            h = h * delta_t.unsqueeze(1) + B_t.unsqueeze(2) * x_t.unsqueeze(1)
            
            # Output: y_t = C_t * sum(h_t)
            y_t = (h.sum(dim=1) * C_t)  # (B, d_inner)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


class MambaBlock(nn.Module):
    """
    A single Mamba block with residual connection and normalization.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand_factor)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        # Pre-normalization
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        
        # Residual connection
        return x + residual


class Mamba(nn.Module):
    """
    Mamba model for sequence classification.
    Uses selective state space models instead of attention.
    """
    def __init__(
        self,
        embedding_type,
        src_vocab_size,
        trg_vocab_size,
        embedding_dim=256,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        num_layers=4,
        dropout=0.1,
        device="cuda",
        max_length=512,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.max_length = max_length
        
        # Embedding layer
        if embedding_type == EmbeddingType.NONE:
            self.embedding = None
        else:
            self.embedding = EmbeddingType.embedding_layer(embedding_type)(
                vocab_size=src_vocab_size,
                embedding_dim=embedding_dim,
                dropout_prob=dropout,
                max_length=max_length,
                device=device
            )
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=embedding_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Classification head
        self.cls = Classifier(
            trg_vocab_size=trg_vocab_size,
            embedding_dim=embedding_dim,
            softmax_dim=2
        )
        
    def forward(self, x, mask=None):
        """
        Forward pass through the Mamba model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) with token indices
            mask: Optional attention mask (not used in Mamba but kept for API compatibility)
        
        Returns:
            Output logits of shape (batch_size, seq_len, trg_vocab_size)
        """
        # Get embeddings
        if self.embedding is not None:
            x = self.embedding(x)
        
        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Classification
        output = self.cls(x)
        
        return output


class MambaEncoder(nn.Module):
    """
    Mamba encoder for sequence-to-sequence tasks.
    Similar to PyTorchTransformerEncoder but using Mamba blocks.
    """
    def __init__(
        self,
        embedding_type,
        src_vocab_size,
        trg_vocab_size,
        embedding_dim=256,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        num_layers=4,
        dropout=0.1,
        device="cuda",
        max_length=512,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.max_length = max_length
        
        # Embedding layer
        if embedding_type == EmbeddingType.NONE:
            self.embedding = None
        else:
            self.embedding = EmbeddingType.embedding_layer(embedding_type)(
                vocab_size=src_vocab_size,
                embedding_dim=embedding_dim,
                dropout_prob=dropout,
                max_length=max_length,
                device=device
            )
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=embedding_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Classification head
        self.cls = Classifier(
            trg_vocab_size=trg_vocab_size,
            embedding_dim=embedding_dim,
            softmax_dim=2
        )
        
    def forward(self, src, mask=None):
        """
        Forward pass through the Mamba encoder.
        
        Args:
            src: Input tensor of shape (batch_size, seq_len) with token indices
            mask: Optional mask (kept for API compatibility)
        
        Returns:
            Output logits of shape (batch_size, seq_len, trg_vocab_size)
        """
        # Get embeddings
        if self.embedding is not None:
            x = self.embedding(src)
        else:
            x = src
        
        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Classification
        output = self.cls(x)
        
        return output
