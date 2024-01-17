# PyTorch Sequence Models

This repository provides models for training deep neural networks on sequential data.

## Sequence to Sequence Modelling with RNNs

You can use this repository for training an encoder-decoder RNN model that can be used for sequence-to-sequence modelling. Typical use cases are:

- Machine translation
- Text summarization
- Chatbots

You can train a model in the following way:

1. Import the necessary modules

```python
import numpy as np
import torch
from dataset.npz_dataset import NPZSequencesDataset
from models.rnn import CellType
from torch.utils.data import DataLoader
from training.seq2seq_rnn_attn import Seq2SeqAttentionRNNPredictor
```

2. Check if GPU is available

```python
# Find out if a CUDA device (GPU) is available
device = "cuda" if torch.cuda.device_count() else "cpu"
```

3. Set hyperparameters

```python
lr = 1e-3                   # The learning rate of the model
cell_type=CellType.LSTM     # Cell type (LSTM | GRU | RNN)
n_epochs = 10               # Number of epochs
num_layers=2                # Number of RNN layers
embedding_dim=32            # Embedding dimension
hidden_size=32              # Hidden size of the RNN layers
batch_size=256              # Batch size used for training
max_length=20               # Maximum sequence length
bidirectional=True          # True if bidirectional RNN layers should be used, False otherwise
```

4. Create a dataset and a dataloader

```python
# Create an instance of the dataset and a dataloader
dataset = ... # Any PyTorch Dataset instance that returns a tensor of shape (batch_size, sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
```

5. Create a predictor instance

```python
predictor = Seq2SeqAttentionRNNPredictor(
    vocab_size_in=vocab_size_en,
    vocab_size_out=vocab_size_fr,
    max_length=max_length,
    num_layers=num_layers,
    batch_size=batch_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    cell_type=cell_type,
    bidirectional=bidirectional,
    device=device,
)
```

6. Train the predictor

```python
predictor.train(
    dataloader=dataloader,
    epochs=n_epochs,
    batch_size=batch_size,
    lr=lr,
)
```
