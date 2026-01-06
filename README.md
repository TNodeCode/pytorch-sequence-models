# PyTorch Sequence Models

This repository provides models for training deep neural networks on sequential data.

## Features

- RNN-based sequence-to-sequence models (LSTM, GRU, RNN)
- Transformer models (Encoder, Decoder, Encoder-Decoder)
- **State Space Models** - Efficient sequence classification with state space models
- **HuggingFace Datasets integration** - Use datasets from the HuggingFace Hub with the models in this repository
- Custom dataset classes for text classification compatible with HuggingFace

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
lr=1e-3                     # The learning rate of the model
cell_type=CellType.LSTM     # Cell type (LSTM | GRU | RNN)
n_epochs=10                 # Number of epochs
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

## State Space Models for Sequence Classification

State Space Models (SSMs) offer an efficient alternative to Transformers and RNNs for sequence modeling. They are based on continuous-time state space equations discretized for sequence processing.

### Key Advantages

- **Efficiency**: Linear-time complexity compared to quadratic for Transformers
- **Long sequences**: Better at handling long-range dependencies than RNNs
- **Simplicity**: Simpler architecture than attention mechanisms

### Usage Example

```python
import torch
from models.state_space import StateSpaceModel
from torch.utils.data import DataLoader

# Set hyperparameters
vocab_size = 10000
num_classes = 5
embedding_dim = 64
d_state = 32
num_layers = 4
max_length = 512
batch_size = 32

# Create the model
model = StateSpaceModel(
    vocab_size=vocab_size,
    num_classes=num_classes,
    embedding_dim=embedding_dim,
    d_state=d_state,
    num_layers=num_layers,
    max_length=max_length,
    pooling='mean',  # Options: 'mean', 'max', 'last', 'cls'
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Prepare your data
dataloader = DataLoader(your_dataset, batch_size=batch_size, shuffle=True)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for sequences, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Pooling Strategies

The State Space Model supports different pooling strategies for sequence classification:

- `'mean'`: Average pooling over the sequence (default)
- `'max'`: Max pooling over the sequence
- `'last'`: Use the last token's representation
- `'cls'`: Use the first token (CLS token style)

### Complete Example

See the `State Space Model Training.ipynb` notebook for a complete example including:
- Model initialization and configuration
- Training on a synthetic dataset
- Evaluation and visualization
- Comparison of different pooling methods

## HuggingFace Datasets Integration

This repository now supports the HuggingFace Datasets library, allowing you to use datasets from the HuggingFace Hub with the models in this repository.

### Using HuggingFace Datasets

You can use any HuggingFace dataset with the models by using the `HuggingFaceDatasetAdapter`:

```python
from datasets import load_dataset
from dataset.huggingface_dataset import HuggingFaceDatasetAdapter
from torch.utils.data import DataLoader

# Load a dataset from HuggingFace Hub
hf_dataset = load_dataset("your_dataset_name", split="train")

# Wrap it with the adapter
adapted_dataset = HuggingFaceDatasetAdapter(
    hf_dataset=hf_dataset,
    input_column='input_ids',
    target_column='labels'
)

# Use with PyTorch DataLoader
dataloader = DataLoader(adapted_dataset, batch_size=32, shuffle=True)
```

### Creating Custom Text Classification Datasets

You can create custom datasets for text classification that are compatible with both PyTorch and HuggingFace:

```python
from dataset.huggingface_dataset import HuggingFaceSequenceClassificationDataset

# Create a custom dataset
dataset = HuggingFaceSequenceClassificationDataset(
    sequences=your_tokenized_sequences,  # List of token ID sequences
    labels=your_labels,                   # List of labels
    max_length=512,
    pad_token_id=0
)

# Convert to HuggingFace format if needed
hf_dataset = dataset.to_huggingface_dataset()
```

### Complete Example

See the `HuggingFace Datasets Example.ipynb` notebook for a complete example of:
- Using the HuggingFace dataset adapter
- Creating custom text classification datasets
- Training models with HuggingFace datasets

