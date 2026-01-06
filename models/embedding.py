import torch
import torch.nn as nn
import torch.nn.functional as F
import math

    
class EmbeddingType:
    NONE="none"
    SINE_COSINE = "sine_cosine"
    POS_LEARNED = "pos_learned"
    
    @staticmethod
    def embedding_layer(embedding_type):
        embedding_types = {
            "none": None,
            "sine_cosine": SineCosineEncoding,
            "pos_learned": LearnedPositionEmbedding,
        }
        return embedding_types[embedding_type]
    

class SineCosineEncoding(nn.Module):
    """
    Classical sine cosine positional encoding.
    @see: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(
            self,
            embedding_dim: int,
            max_length: int = 5000
        ):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_length, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self) -> torch.Tensor:
        """
        Return:
            sine cosine positional encoding matrix
        """
        return self.pe[:self.max_length]


class PositionEncodingEmbedding(nn.Module):
    """
    Module that creates an embedding by adding the word embedding and the sine-cosine position encoding
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_length: int,
        dropout_prob: float=0.1,
        device: str='cpu',
        **kwargs
    ):
        """    
        Keyword arguments:
        vocab_size    -- The number of unique tokens in the vocabulary
        embedding_dim -- dimension of the embedding that this function will output
        max_length    -- Maximum length of the sequence
        dropout_prob  -- probability of decativating neurons randomly
        device        -- Device that the operations should run on (cpu | cuda)
        """        
        super(PositionEncodingEmbedding, self).__init__()
        
        # Hyperparameters
        self.device = device
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        
        # Layers
        self.position_embedding = SineCosineEncoding(embedding_dim=embedding_dim, max_length=max_length)
        self.word_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_prob)
        
    def forward(self, x: int, pos: int):
        """
        Create an embedding that contains the embedded input word and the coordinates
        
        Keyword arguments:
        x   -- The indices of the tokens in the input sequence
        pos -- the positions (integers from 1 to max_length) of the input tokens

        Returns:
        This functions returns an embedding that contains the information aboout a token and its position        
        """
        # Run the input token through the embedding layer and then add the position encoding to it
        return self.dropout(
            self.word_embedding(x) +
            self.position_embedding.forward()
        )


class LearnedPositionEmbedding(nn.Module):
    """
    Module that learnes a word embedding and a position embedding and adds them
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_length: int,
        dropout_prob: float=0.1,
        device: str='cpu',
        **kwargs
    ):
        """    
        Keyword arguments:
        vocab_size    -- The number of unique tokens in the vocabulary
        embedding_dim -- dimension of the embedding that this function will output
        dropout_prob  -- probability of decativating neurons randomly
        device        -- Device that the operations should run on (cpu | cuda)
        """        
        super(LearnedPositionEmbedding, self).__init__()
        
        # Hyperparameters
        self.device = device
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        
        # Layers
        self.word_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = torch.nn.Embedding(max_length, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_prob)
        
    def forward(self, x, **kwargs):
        """
        Create an embedding that contains the embedded input word and the embedded position
        
        Keyword arguments:
        x           -- The index of the token in the sequence

        Returns:
        This functions returns an embedding that contains the information aboout a token and its position        
        """
        # Extract batch size and sequence length from input
        batch_size, seq_len = x.shape
        # Create position array
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # First run the input sequences through an embedding layer
        embedded_word = self.dropout(self.word_embedding(x))
        # Then run the positions through an embedding layer
        embedded_pos = self.dropout(self.pos_embedding(pos))
        # Concatenate word embeddings with position embeddings
        return embedded_word + embedded_pos