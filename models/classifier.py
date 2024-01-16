import torch.nn as nn


class Classifier(nn.Module):
    """
    This module takes the outputs of a decoder and runs them through a classification network
    that consists of linear layers and a final softmax classification layer.
    """
    def __init__(
        self,
        trg_vocab_size,
        embedding_dim,
        hidden_size=256,
        softmax_dim=2,
    ):
        super(Classifier, self).__init__()
        
        # Hyperparameters
        self.trg_vocab_size = trg_vocab_size
        self.embedding_dim = embedding_dim
        
        # Layers
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=softmax_dim)
        
    def forward(self, x):
        return self.softmax(self.fc2(self.fc1(x)))