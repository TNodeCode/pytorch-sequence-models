import numpy as np
from torch.utils.data import Dataset


class NPZSequencesDataset(Dataset):
    """
    This class loads data from a npz file
    """
    def __init__(
            self,
            input_seqs_filename: str,
            output_seqs_filename: str,
            key="data",
            max_length=50,
            split='train',
            split_train=0.8,
            split_val=0.1,
            split_test=0.1,
    ):
        """
        Initialize the dataset class
        :param filename: The filename of the CSV file
        """
        self.source_seqs = np.load(input_seqs_filename)[key][:, :max_length]
        self.target_seqs = np.load(output_seqs_filename)[key][:, :max_length]            
        self.vocab_in_size = self.source_seqs.max() + 1
        self.vocab_out_size = self.target_seqs.max() + 1
        assert split_train + split_val + split_test, "Split sizes must add to one"
        if split == 'train':
            start, end = 0, int(len(self.source_seqs) * split_train)
        elif split == 'val':
            start, end = int(len(self.source_seqs) * split_train), int(len(self.source_seqs) * (split_train + split_val))
        elif split == 'test':
            start, end = int(len(self.source_seqs) * (split_train + split_val)), len(self.source_seqs)
        else:
            raise ValueError(f"Split {split} is not defined")
        self.source_seqs = self.source_seqs[start:end]
        self.target_seqs = self.target_seqs[start:end]
        if self.source_seqs.shape[0] != self.target_seqs.shape[0]:
            raise Exception("Number of samples of source and target sequences must be equal")

    def __len__(self):
        """
        This function returns the total number of items in the dataset.
        We are using a pandas data frame in this dataset which has an attribut named shape.
        The first dimension of shape is equal to the number of items in the dataset.
        :return: The number of rows in the CSV file
        """
        return self.source_seqs.shape[0]

    def __getitem__(self, idx):
        """
        This function returns a single tuple from the dataset.
        :param idx: The index of the tuple that should be returned.
        :return: Tuple of an x-value and a y-value
        """
        return self.source_seqs[idx], self.target_seqs[idx]