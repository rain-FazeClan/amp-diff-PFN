import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from utils import vocab, MAX_LEN, PAD_TOKEN, AMINO_ACIDS # Import shared utilities


class GANPeptideDataset(Dataset):
    def __init__(self, classify_csv_path, vocab, max_len):
        """
        Dataset for GAN training, loads sequence and label from featurized data.
        """
        # Read the featurized data, but we only need Sequence and label for GAN
        self.data = pd.read_csv(classify_csv_path)

        # Filter out sequences longer than max_len if they exist
        self.data = self.data[self.data['sequence'].str.len() <= max_len].copy()

        self.sequences = self.data['sequence'].tolist()
        self.labels = self.data['label'].tolist()
        self.vocab = vocab
        self.max_len = max_len

        # Encode and pad sequences
        self.encoded_sequences = [self.vocab.encode(seq) for seq in self.sequences]
        self.padded_sequences = [
            self.vocab.pad_sequence(seq) for seq in self.encoded_sequences
        ]
        self.padded_sequences = torch.tensor(self.padded_sequences, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Return padded sequence and its original label (0 for neg, 1 for pos)
        return self.padded_sequences[idx], self.labels[idx]

def create_gan_dataloader(batch_size, shuffle=True):
    """
    Creates DataLoader for GAN training.
    """
    classify_csv_path = os.path.join('classify_data/classify.csv')
    if not os.path.exists(classify_csv_path):
        raise FileNotFoundError(f"Required data file not found: {classify_csv_path}. Please run featured_data_generated.py first.")

    dataset = GANPeptideDataset(classify_csv_path, vocab, MAX_LEN)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"Loaded {len(dataset)} sequences for GAN training.")
    return dataloader, dataset # Also return dataset for potential future use

if __name__ == '__main__':
    # Example usage
    BATCH_SIZE = 64
    dataloader, _ = create_gan_dataloader(BATCH_SIZE)
    for batch_seqs, batch_labels in dataloader:
        print("Sample Batch:")
        print("Sequences (indices):", batch_seqs.shape, batch_seqs.dtype)
        print("Labels:", batch_labels.shape, batch_labels.dtype)
        # print("Decoded sequence example:", vocab.decode(batch_seqs[0].tolist())) # Uncomment if you want to decode one example
        break