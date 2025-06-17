# Shared Constants
MAX_LEN = 25 # Max peptide length for GAN
VOCABULARY = 'ACDEFGHIKLMNPQRSTVWY' # 20 standard amino acids
PAD_TOKEN = '<PAD>'
# Optional START/END tokens if needed for variable length or generation start
AMINO_ACIDS = [PAD_TOKEN] + list(VOCABULARY)

# Discriminator classes
# 0: Real Negative (non-AMP)
# 1: Real Positive (AMP)
# 2: Fake (Generated)
NUM_DISC_CLASSES = 3


class Vocab:
    def __init__(self, amino_acids, pad_token=PAD_TOKEN):
        self.amino_acids = amino_acids
        self.pad_token = pad_token
        self.word_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        self.idx_to_word = {i: aa for i, aa in enumerate(amino_acids)}
        self.pad_idx = self.word_to_idx[pad_token]
        self.vocab_size = len(amino_acids)

    def encode(self, sequence):
        encoded = [self.word_to_idx.get(aa, self.pad_idx) for aa in sequence]
        return encoded

    def decode(self, indices, remove_padding=True):
        words = [self.idx_to_word[idx] for idx in indices]
        decoded_seq = "".join(words)
        if remove_padding:
             decoded_seq = decoded_seq.rstrip(self.pad_token)
        return decoded_seq

    def pad_sequence(self, sequence_indices):
         if len(sequence_indices) > MAX_LEN:
             padded = sequence_indices[:MAX_LEN]
         else:
             padded = sequence_indices + [self.pad_idx] * (MAX_LEN - len(sequence_indices))
         return padded

# Global Vocab instance
vocab = Vocab(AMINO_ACIDS)