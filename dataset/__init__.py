"""
Dataset module for pytorch-sequence-models.

This module contains dataset classes for loading and processing data
for sequence models.
"""

from dataset.npz_dataset import NPZSequencesDataset
from dataset.huggingface_dataset import (
    HuggingFaceDatasetAdapter,
    HuggingFaceSequenceClassificationDataset
)

__all__ = [
    'NPZSequencesDataset',
    'HuggingFaceDatasetAdapter',
    'HuggingFaceSequenceClassificationDataset'
]
