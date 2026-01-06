import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Any


class HuggingFaceDatasetAdapter(Dataset):
    """
    Adapter class that wraps a HuggingFace dataset to make it compatible 
    with PyTorch models in this repository.
    
    This adapter converts HuggingFace dataset items into tensors that can be
    used with the sequence models defined in this repository.
    """
    
    def __init__(
        self,
        hf_dataset,
        input_column: str = "input_ids",
        target_column: Optional[str] = "labels",
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the HuggingFace dataset adapter.
        
        Args:
            hf_dataset: A HuggingFace dataset object (from datasets library)
            input_column: Name of the column containing input sequences
            target_column: Name of the column containing target sequences/labels (optional)
            transform: Optional transform to apply to the data
        """
        self.hf_dataset = hf_dataset
        self.input_column = input_column
        self.target_column = target_column
        self.transform = transform
        
    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        """
        Returns a single item from the dataset.
        
        Args:
            idx: The index of the item to return
            
        Returns:
            If target_column is provided: tuple of (input_tensor, target_tensor)
            If target_column is None: input_tensor only
        """
        item = self.hf_dataset[idx]
        
        # Get input data
        input_data = item[self.input_column]
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data)
        
        # Apply transform if provided
        if self.transform:
            input_data = self.transform(input_data)
        
        # If target column is specified, return both input and target
        if self.target_column is not None:
            target_data = item[self.target_column]
            if not isinstance(target_data, torch.Tensor):
                target_data = torch.tensor(target_data)
            return input_data, target_data
        
        # Otherwise, return only input
        return input_data
    
    @property
    def vocab_in_size(self):
        """
        Returns the input vocabulary size if available.
        
        Note: This works automatically for classification features with num_classes.
        For sequence features (token IDs), this returns None by default.
        If needed, compute vocabulary size manually by finding max(token_ids) + 1
        across your dataset.
        """
        if hasattr(self.hf_dataset, 'features'):
            if self.input_column in self.hf_dataset.features:
                feature = self.hf_dataset.features[self.input_column]
                # For classification features
                if hasattr(feature, 'num_classes'):
                    return feature.num_classes
        return None
    
    @property
    def vocab_out_size(self):
        """
        Returns the output vocabulary size if available.
        
        Note: This works automatically for classification features with num_classes.
        For sequence features (token IDs), this returns None by default.
        If needed, compute vocabulary size manually by finding max(token_ids) + 1
        across your dataset.
        """
        if self.target_column and hasattr(self.hf_dataset, 'features'):
            if self.target_column in self.hf_dataset.features:
                feature = self.hf_dataset.features[self.target_column]
                # For classification features
                if hasattr(feature, 'num_classes'):
                    return feature.num_classes
        return None


class HuggingFaceSequenceClassificationDataset(Dataset):
    """
    A custom dataset class for sequence classification tasks that is compatible
    with both PyTorch and HuggingFace datasets library.
    
    This dataset can be used for text classification tasks where you have
    sequences of tokens (as indices) and corresponding labels.
    """
    
    def __init__(
        self,
        sequences,
        labels,
        max_length: int = 512,
        pad_token_id: int = 0,
    ):
        """
        Initialize the sequence classification dataset.
        
        Args:
            sequences: List or array of token sequences (each sequence is a list of token ids)
            labels: List or array of labels corresponding to each sequence
            max_length: Maximum sequence length (sequences will be truncated or padded)
            pad_token_id: Token ID to use for padding
        """
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        
        if len(self.sequences) != len(self.labels):
            raise ValueError("Number of sequences must match number of labels")
    
    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns a single item from the dataset.
        
        Args:
            idx: The index of the item to return
            
        Returns:
            Dictionary with 'input_ids' and 'labels' keys
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert to list if necessary
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        elif not isinstance(sequence, list):
            sequence = list(sequence)
        
        # Truncate or pad sequence
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [self.pad_token_id] * (self.max_length - len(sequence))
        
        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def to_huggingface_dataset(self):
        """
        Convert this dataset to a HuggingFace Dataset object.
        
        Returns:
            A HuggingFace Dataset object
        """
        try:
            from datasets import Dataset as HFDataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required to convert to HuggingFace format. "
                "Install it with: pip install datasets"
            )
        
        # Prepare data in dictionary format
        data_dict = {
            'input_ids': [],
            'labels': []
        }
        
        for i in range(len(self)):
            item = self[i]
            data_dict['input_ids'].append(item['input_ids'].tolist())
            data_dict['labels'].append(item['labels'].item())
        
        return HFDataset.from_dict(data_dict)
