import torch
import torch.nn as nn
from models.embedding import EmbeddingType
from models.classifier import Classifier


class PyTorchTransformer(nn.Module):
    """
    Encoder-Decoder Transformer model for sequence-to-sequence tasks.
    
    This model uses PyTorch's built-in Transformer implementation with custom embeddings
    and a classification head. Suitable for tasks like machine translation where both
    input and output sequences are processed.
    
    Args:
        embedding_type: Type of embedding to use (EmbeddingType.POS_LEARNED, EmbeddingType.SINE_COSINE, or EmbeddingType.NONE)
        src_vocab_size: Size of source vocabulary
        trg_vocab_size: Size of target vocabulary
        embedding_dim: Dimension of embeddings (default: 256)
        num_layers: Number of encoder and decoder layers (default: 6)
        dim_feedforward: Dimension of feedforward network (default: 2048)
        heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        device: Device to run model on (default: "cuda")
        max_length: Maximum sequence length (default: 100)
    """
    def __init__(
        self,
        embedding_type,
        src_vocab_size,
        trg_vocab_size,
        embedding_dim=256,
        num_layers=6,
        dim_feedforward=2048,
        heads=8,
        dropout=0.1,
        device="cuda",
        max_length=100,
        **kwargs,
    ):
        super(PyTorchTransformer, self).__init__()
        self.device = device
        self.embedding_type = embedding_type
        
        # Layers
        if embedding_type == EmbeddingType.NONE:
            self.encoder_embedding = None
        else:
            self.encoder_embedding = EmbeddingType.embedding_layer(embedding_type)(
                vocab_size=src_vocab_size,
                embedding_dim=embedding_dim,
                dropout_prob=dropout,
                max_length=max_length,
                device=device            
            )
        if embedding_type == EmbeddingType.NONE:
            self.decoder_embedding = None
        else:
            self.decoder_embedding = EmbeddingType.embedding_layer(embedding_type)(
                vocab_size=trg_vocab_size,
                embedding_dim=embedding_dim,
                dropout_prob=dropout,
                max_length=max_length,
                device=device            
            )
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.cls = Classifier(
            trg_vocab_size=trg_vocab_size,
            embedding_dim=embedding_dim,
            softmax_dim=2
        )
        
    def forward(self, src, trg, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Get input sequence embeddings
        if self.encoder_embedding is not None:
            src_embedding = self.encoder_embedding(src)
        else:
            src_embedding = src
        # Get target sequence embeddings
        if self.decoder_embedding is not None:
            trg_embedding = self.decoder_embedding(trg)
        else:
            trg_embedding = trg
        # Run the embeddings through the transformer
        out = self.transformer(
            src_embedding,
            trg_embedding,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        # Run outputs of the transformer through the classification network
        out = self.cls(out)
        return out
    

class PyTorchTransformerEncoder(nn.Module):
    """
    Transformer Encoder model for sequence classification tasks.
    
    This model uses PyTorch's TransformerEncoder with custom embeddings and a classification head.
    Suitable for tasks like text classification, sentiment analysis, or sequence tagging where
    only the input sequence needs to be encoded.
    
    Args:
        embedding_type: Type of embedding to use (EmbeddingType.POS_LEARNED, EmbeddingType.SINE_COSINE, or EmbeddingType.NONE)
        src_vocab_size: Size of source vocabulary
        trg_vocab_size: Size of target vocabulary (output classes)
        embedding_dim: Dimension of embeddings (default: 256)
        dim_feedforward: Dimension of feedforward network (default: 2048)
        num_layers: Number of encoder layers (default: 6)
        heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        device: Device to run model on (default: "cuda")
        max_length: Maximum sequence length (default: 100)
    """
    def __init__(
        self,
        embedding_type,
        src_vocab_size,
        trg_vocab_size,
        embedding_dim=256,
        dim_feedforward=2048,
        num_layers=6,
        heads=8,
        dropout=0.1,
        device="cuda",
        max_length=100,
        **kwargs,
    ):
        super(PyTorchTransformerEncoder, self).__init__()
        self.device = device
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.device = device
        self.max_length = max_length
        
        # Layers
        if embedding_type == EmbeddingType.NONE:
            self.encoder_embedding = None
        else:
            self.encoder_embedding = EmbeddingType.embedding_layer(embedding_type)(
                vocab_size=src_vocab_size,
                embedding_dim=embedding_dim,
                dropout_prob=dropout,
                max_length=max_length,
                device=device            
            )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=self.num_layers
        )
        self.cls = Classifier(
            trg_vocab_size=trg_vocab_size,
            embedding_dim=embedding_dim,
            softmax_dim=2
        )
        
    def forward(self, src, mask=None):
        # Get input sequence embeddings
        if self.encoder_embedding is not None:
            src_embedding = self.encoder_embedding(src)
        else:
            src_embedding = src
        # Run the embeddings through the transformer
        out = self.encoder.forward(
            src=src_embedding,
            mask=mask,
        )
        # Run outputs of the transformer through the classification network
        out = self.cls(out)
        return out


class PyTorchTransformerDecoder(nn.Module):
    """
    Transformer Decoder model for sequence generation tasks.
    
    This model uses PyTorch's TransformerDecoder with custom embeddings and a classification head.
    The decoder attends to encoded memory (source) while generating the target sequence.
    Suitable for tasks where you need autoregressive decoding conditioned on source input.
    
    Args:
        embedding_type: Type of embedding to use (EmbeddingType.POS_LEARNED, EmbeddingType.SINE_COSINE, or EmbeddingType.NONE)
        src_vocab_size: Size of source vocabulary
        trg_vocab_size: Size of target vocabulary
        embedding_dim: Dimension of embeddings (default: 256)
        dim_feedforward: Dimension of feedforward network (default: 2048)
        num_layers: Number of decoder layers (default: 6)
        heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        device: Device to run model on (default: "cuda")
        max_length: Maximum sequence length (default: 100)
    """
    def __init__(
        self,
        embedding_type,
        src_vocab_size,
        trg_vocab_size,
        embedding_dim=256,
        dim_feedforward=2048,
        num_layers=6,
        heads=8,
        dropout=0.1,
        device="cuda",
        max_length=100,
        **kwargs,
    ):
        super(PyTorchTransformerDecoder, self).__init__()
        self.device = device
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.device = device
        self.max_length = max_length
        
        # Layers
        if embedding_type == EmbeddingType.NONE:
            self.encoder_embedding = None
        else:
            self.encoder_embedding = EmbeddingType.embedding_layer(embedding_type)(
                vocab_size=src_vocab_size,
                embedding_dim=embedding_dim,
                dropout_prob=dropout,
                max_length=max_length,
                device=device            
            )
        if embedding_type == EmbeddingType.NONE:
            self.decoder_embedding = None
        else:
            self.decoder_embedding = EmbeddingType.embedding_layer(embedding_type)(
                vocab_size=trg_vocab_size,
                embedding_dim=embedding_dim,
                dropout_prob=dropout,
                max_length=max_length,
                device=device            
            )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=self.heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=self.num_layers
        )
        self.cls = Classifier(
            trg_vocab_size=trg_vocab_size,
            embedding_dim=embedding_dim,
            softmax_dim=2
        )
        
    def forward(self, src, trg, src_mask=None, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Get input sequence embeddings
        if self.encoder_embedding is not None:
            src_embedding = self.encoder_embedding(src)
        else:
            src_embedding = src
        # Get target sequence embeddings
        if self.decoder_embedding is not None:
            trg_embedding = self.decoder_embedding(trg)
        else:
            trg_embedding = trg
        # Run the embeddings through the transformer
        out = self.decoder.forward(
            tgt=trg_embedding,
            memory=src_embedding,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        # Run outputs of the transformer through the classification network
        out = self.cls(out)
        return out