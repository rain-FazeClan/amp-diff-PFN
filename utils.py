# src/utils.py (复用或稍微调整)
import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYXU*" # X:未知，U:Seleno Cysteine, *:终止符
# We will use 0 as the padding value

def amino_acid_to_int(amino_acid, padding_char=None, padding_value=0):
    """将氨基酸字符映射为整数索引 (padding_value保留作填充符)"""
    if padding_char is None:
        # Determine padding_char from padding_value if needed
        # Assuming padding_value=0 is used as an index that won't map to an actual AA in AMINO_ACIDS if we re-index
        pass # For simplicity, let's map based on AMINO_ACIDS order, index 0 corresponds to 'A'
    try:
        # Check if the character is part of our defined alphabet
        return AMINO_ACIDS.index(amino_acid)
    except ValueError:
        # If unknown, return index for 'X'
        return AMINO_ACIDS.index('X')

# Let's redefine mapping to ensure 0 is only padding if intended
# A better approach might be to have a dedicated PADDING_VALUE constant and handle it explicitly.
PADDING_VALUE = 0 # Use 0 as padding index

# Create mapping from character to index, indices starting from 1 for actual amino acids
# And mapping from index back to character.
# This ensures 0 is clearly only padding.
AMINO_ACID_CHARS = list(AMINO_ACIDS) # Convert string to list of chars
AMINO_ACID_TO_INT_MAP = {char: i + 1 for i, char in enumerate(AMINO_ACID_CHARS)}
INT_TO_AMINO_ACID_MAP = {i + 1: char for i, char in enumerate(AMINO_ACID_CHARS)}
# Add padding mapping
INT_TO_AMINO_ACID_MAP[PADDING_VALUE] = '' # Or a special token like '<PAD>'


def sequence_to_int_sequence(sequence, padding_value=PADDING_VALUE):
    """将氨基酸序列转换为整数序列"""
    # Use the new map. Filter out characters not in our alphabet before mapping.
    # Treat unknown characters 'X' explicitly.
    int_seq = []
    for aa in sequence:
        if aa in AMINO_ACID_TO_INT_MAP:
            int_seq.append(AMINO_ACID_TO_INT_MAP[aa])
        elif aa == INT_TO_AMINO_ACID_MAP.get(padding_value): # If it's the padding character representation
             continue # Skip padding chars if they somehow end up in the input sequences
        else:
            # Handle unknown character - map to 'X'
            int_seq.append(AMINO_ACID_TO_INT_MAP['X'])
    return int_seq

def int_sequence_to_sequence(int_sequence, padding_value=PADDING_VALUE, stop_char='*'):
    """将整数序列转换为氨基酸序列 (不包含填充符和停止符之后的字符)"""
    seq = []
    for i in int_sequence:
        if i == padding_value:
            continue # Skip padding value
        aa = INT_TO_AMINO_ACID_MAP.get(i, 'X') # Map index back to character, default to 'X' if unknown index
        if aa == stop_char:
            break # Stop at the stop character
        seq.append(aa)
    return "".join(seq)


def pad_sequences(sequences, max_length, padding='post', value=PADDING_VALUE):
    """对整数序列进行填充"""
    padded_sequences = np.full((len(sequences), max_length), value, dtype=np.int64) # Use int64 for PyTorch compatibility
    for i, seq in enumerate(sequences):
        seq_array = np.array(seq, dtype=np.int64)
        current_len = len(seq_array)
        if current_len > max_length: # Truncate if too long
            if padding == 'post':
                padded_sequences[i, :] = seq_array[:max_length]
            else: # 'pre' padding, truncate from the beginning
                 padded_sequences[i, :] = seq_array[-max_length:]
            # print(f"警告: 序列被截断 from {current_len} to {max_length}") # Optional warning
        else:
            if padding == 'post':
                padded_sequences[i, :current_len] = seq_array
            elif padding == 'pre':
                padded_sequences[i, max_length - current_len:] = seq_array
    return padded_sequences

# Need the actual number of characters used for one-hot encoding (excluding padding)
NUM_AMINO_ACIDS = len(AMINO_ACID_CHARS) # Total chars in the alphabet
NUM_AMINO_ACIDS_ONE_HOT = NUM_AMINO_ACIDS # Number of columns in one-hot vector (corresponds to indices 1 to NUM_AMINO_ACIDS)


def one_hot_encode_sequence(int_sequence, num_chars_for_one_hot=NUM_AMINO_ACIDS_ONE_HOT, padding_value=PADDING_VALUE):
    """
    对单个整数序列进行One-Hot编码。

    Args:
        int_sequence (list or np.ndarray): 整数编码的序列 (Already padded).
        num_chars_for_one_hot (int): One-Hot 向量的维度（词汇表大小）。
        padding_value (int): 填充使用的整数值。

    Returns:
        np.ndarray: One-Hot 编码后的序列 (sequence_length, num_chars_for_one_hot).
                   Padding positions will be all zeros.
    """
    # Ensure indices are within range [1, num_chars_for_one_hot] or are the padding_value
    # The one-hot encoding will have dimensions up to num_chars_for_one_hot, indexed 0 to num_chars_for_one_hot-1
    # Our current mapping uses indices 1 to NUM_AMINO_ACIDS for characters.
    # So we need num_chars_for_one_hot to be NUM_AMINO_ACIDS + 1 to include index 0 for padding
    # Or, use vocab size = NUM_AMINO_ACIDS + 1, and hot-encode indices 1 to NUM_AMINO_ACIDS, index 0 is all zero vector
    vocab_size_for_one_hot = NUM_AMINO_ACIDS + 1 # Vocabulary includes indices 0 (padding) up to NUM_AMINO_ACIDS
    one_hot = np.zeros((len(int_sequence), vocab_size_for_one_hot), dtype=np.float32)
    for i, seq_int in enumerate(int_sequence):
        if seq_int != padding_value and seq_int > 0 and seq_int <= NUM_AMINO_ACIDS:
             # Indices need to be 0-based for typical one-hot indexing.
             # If seq_int is 1-based (A=1, C=2 etc), map it to 0-based (A=0, C=1 etc) for one-hot columns.
             # So, one-hot column index is seq_int - 1
            one_hot[i, seq_int - 1] = 1.0 # Map 1-based index to 0-based one-hot column
        elif seq_int == padding_value:
            # Padding value (0) results in an all-zero vector, which is correctly initialized above.
            pass
        else:
             # Handle unexpected indices, map to 'X' one-hot or all zeros. Let's map to 'X' index (which is last one-hot column)
             print(f"警告: 发现意外的整数编码 {seq_int}, 将其映射到 'X' one-hot 向量.")
             x_index = AMINO_ACID_TO_INT_MAP['X'] # Get the 1-based index for 'X'
             one_hot[i, x_index - 1] = 1.0 # Map 1-based 'X' index to 0-based column

    # Decide whether to return num_chars_for_one_hot columns or vocab_size_for_one_hot columns.
    # If Discriminator expects (batch, max_len, NUM_AMINO_ACIDS), we slice the result
    return one_hot[:, :NUM_AMINO_ACIDS] # Return only the columns for actual amino acids


def sequences_to_one_hot(int_sequences, max_length, num_chars_for_one_hot=NUM_AMINO_ACIDS_ONE_HOT, padding='post', padding_value=PADDING_VALUE):
     """
     对多个整数序列进行 One-Hot 编码和填充。

     Args:
        int_sequences (list of list or np.ndarray): 整数编码的序列列表 (can be padded or not).
        max_length (int): 填充到的最大长度。
        num_chars_for_one_hot (int): One-Hot 向量的维度（词汇表大小不含padding index的维度）。
        padding (str, optional): 填充方式 ('pre' 或 'post')。
        padding_value (int, optional): 填充使用的整数值。

    Returns:
        np.ndarray: One-Hot 编码并填充后的序列批次 (batch_size, max_length, num_chars_for_one_hot)。
     """
     # Ensure input sequences are padded first if they are not already
     if not isinstance(int_sequences, np.ndarray) or int_sequences.shape[1] != max_length:
         int_sequences = pad_sequences(int_sequences, max_length, padding=padding, value=padding_value)

     # Now, int_sequences is a numpy array of shape (batch_size, max_length)
     batch_size = int_sequences.shape[0]
     max_len = int_sequences.shape[1]

     # Allocate numpy array for one-hot output
     one_hot_batch = np.zeros((batch_size, max_len, num_chars_for_one_hot), dtype=np.float32)

     for i in range(batch_size):
         # Encode each padded integer sequence
         one_hot_batch[i] = one_hot_encode_sequence(int_sequences[i], num_chars_for_one_hot, padding_value)

     return one_hot_batch

# --- Example Usage Check (Optional, for debugging utils) ---
# print("Testing utils...")
# seq_str = "ACDEF"
# int_seq = sequence_to_int_sequence(seq_str)
# print(f"Sequence '{seq_str}' -> Int: {int_seq}") # Should be [1, 2, 3, 4, 5]

# padded_int_seq = pad_sequences([int_seq], max_length=10, padding='post')
# print(f"Padded Int: {padded_int_seq}") # Should be [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]]

# recovered_seq = int_sequence_to_sequence(padded_int_seq[0])
# print(f"Padded Int back to Seq: '{recovered_seq}'") # Should be 'ACDEF'

# one_hot_seq = one_hot_encode_sequence(padded_int_seq[0], NUM_AMINO_ACIDS)
# print(f"One-Hot shape: {one_hot_seq.shape}") # Should be (10, NUM_AMINO_ACIDS)
# # print("Sample One-Hot:", one_hot_seq[:2, :5])

# print("-" * 20)