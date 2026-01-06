import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding import EmbeddingType
from models.classifier import Classifier


class StateSpaceLayer(nn.Module):
    """
    A single State Space Model layer implementing the discrete-time state space equations:
    h_t = A * h_t-1 + B * x_t
    y_t = C * h_t + D * x_t
    
    This is a simplified implementation suitable for sequence modeling.
    """
    def __init__(self, d_model, d_state=64, dropout=0.1):
        super(StateSpaceLayer, self).__init__()
        self.d_model = d_model  # Input/output dimension
        self.d_state = d_state  # State dimension
        
        # State space parameters
        # A: State transition matrix (d_state x d_state)
        # We use a diagonal structure for efficiency
        self.A_diag = nn.Parameter(torch.randn(d_state))
        
        # B: Input to state matrix (d_model x d_state)
        # Shape allows: h = h + x @ B => (batch, d_state) = (batch, d_state) + (batch, d_model) @ (d_model, d_state)
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        
        # C: State to output matrix (d_state x d_model)
        # Shape allows: y = h @ C => (batch, d_model) = (batch, d_state) @ (d_state, d_model)
        self.C = nn.Parameter(torch.randn(d_state, d_model))
        
        # D: Skip connection (d_model x d_model)
        self.D = nn.Parameter(torch.randn(d_model, d_model))
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize state space parameters with sensible defaults."""
        # Initialize A to be stable (diagonal elements < 1)
        nn.init.uniform_(self.A_diag, -0.5, 0.5)
        
        # Initialize B and C with Xavier initialization
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        
        # Initialize D to small values
        nn.init.xavier_uniform_(self.D)
        self.D.data *= 0.1
    
    def forward(self, x):
        """
        Forward pass through the state space layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        
        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, d_model)
            
            # State update: h_t = A * h_t-1 + B^T * x_t
            # h: (batch, d_state), x_t: (batch, d_model), B: (d_model, d_state)
            h = torch.sigmoid(self.A_diag) * h + torch.matmul(x_t, self.B)
            
            # Output: y_t = C^T * h_t + D^T * x_t
            # h: (batch, d_state), C: (d_state, d_model)
            y_t = torch.matmul(h, self.C) + torch.matmul(x_t, self.D)
            
            outputs.append(y_t)
        
        # Stack outputs along sequence dimension
        output = torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_model)
        
        return self.dropout(output)


class StateSpaceBlock(nn.Module):
    """
    A State Space block with residual connection and layer normalization.
    Similar to a Transformer block but using State Space layer instead of attention.
    """
    def __init__(self, d_model, d_state=64, d_ff=None, dropout=0.1):
        super(StateSpaceBlock, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # State space layer
        self.ssm = StateSpaceLayer(d_model, d_state, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # State space layer with residual connection
        x = x + self.ssm(self.ln1(x))
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.ln2(x))
        
        return x


class StateSpaceModel(nn.Module):
    """
    State Space Model for sequence classification.
    
    This model stacks multiple State Space blocks and uses a classifier
    on top for sequence-level or token-level classification.
    """
    def __init__(
        self,
        vocab_size,
        num_classes,
        embedding_dim=256,
        d_state=64,
        num_layers=4,
        d_ff=None,
        dropout=0.1,
        max_length=512,
        pooling='mean',
        use_embedding=True,
        device='cpu',
    ):
        """
        Args:
            vocab_size: Size of the vocabulary
            num_classes: Number of output classes
            embedding_dim: Dimension of embeddings and model
            d_state: Dimension of the state space
            num_layers: Number of State Space blocks
            d_ff: Dimension of feed-forward network (default: 4 * embedding_dim)
            dropout: Dropout probability
            max_length: Maximum sequence length
            pooling: Pooling strategy for sequence classification ('mean', 'max', 'last', 'cls')
            use_embedding: Whether to use embedding layer
            device: Device to run the model on
        """
        super(StateSpaceModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.d_state = d_state
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_length = max_length
        self.pooling = pooling
        self.use_embedding = use_embedding
        self.device = device
        
        # Embedding layer
        if use_embedding:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            # Position embeddings (optional but helpful)
            self.position_embedding = nn.Embedding(max_length, embedding_dim)
        else:
            self.embedding = None
            self.position_embedding = None
        
        # State Space blocks
        self.blocks = nn.ModuleList([
            StateSpaceBlock(
                d_model=embedding_dim,
                d_state=d_state,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(embedding_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes)
        )
        
    def _pool_sequence(self, x, mask=None):
        """
        Pool sequence representations into a single vector.
        
        Args:
            x: Sequence tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Pooled tensor of shape (batch_size, embedding_dim)
        """
        if self.pooling == 'mean':
            if mask is not None:
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).float()
                sum_embeddings = (x * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return x.mean(dim=1)
        elif self.pooling == 'max':
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            return x.max(dim=1)[0]
        elif self.pooling == 'last':
            if mask is not None:
                # Get the last non-masked position for each sequence
                lengths = mask.sum(dim=1) - 1
                return x[torch.arange(x.size(0), device=x.device), lengths]
            else:
                return x[:, -1, :]
        elif self.pooling == 'cls':
            # Use the first token (CLS token)
            return x[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def forward(self, x, mask=None):
        """
        Forward pass through the State Space Model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) for token indices
               or (batch_size, seq_len, embedding_dim) for embedded inputs
            mask: Optional mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len = x.shape[:2]
        
        # Embedding
        if self.use_embedding and self.embedding is not None:
            # Token embeddings
            x = self.embedding(x)
            
            # Add position embeddings
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.position_embedding(positions)
        
        # Pass through State Space blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Pool sequence
        x = self._pool_sequence(x, mask)
        
        # Classify
        logits = self.classifier(x)
        
        return logits


class StateSpaceEncoder(nn.Module):
    """
    State Space Encoder for sequence-to-sequence tasks.
    Similar to Transformer Encoder but using State Space layers.
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        d_state=64,
        num_layers=4,
        d_ff=None,
        dropout=0.1,
        max_length=512,
        use_embedding=True,
        device='cpu',
    ):
        """
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of embeddings and model
            d_state: Dimension of the state space
            num_layers: Number of State Space blocks
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
            max_length: Maximum sequence length
            use_embedding: Whether to use embedding layer
            device: Device to run the model on
        """
        super(StateSpaceEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_state = d_state
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_length = max_length
        self.use_embedding = use_embedding
        self.device = device
        
        # Embedding layer
        if use_embedding:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.position_embedding = nn.Embedding(max_length, embedding_dim)
        else:
            self.embedding = None
            self.position_embedding = None
        
        # State Space blocks
        self.blocks = nn.ModuleList([
            StateSpaceBlock(
                d_model=embedding_dim,
                d_state=d_state,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(embedding_dim)
    
    def forward(self, x, mask=None):
        """
        Forward pass through the State Space Encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            mask: Optional mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Encoded tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = x.shape[:2]
        
        # Embedding
        if self.use_embedding and self.embedding is not None:
            x = self.embedding(x)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.position_embedding(positions)
        
        # Pass through State Space blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        return x
