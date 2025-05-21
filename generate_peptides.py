# generate_peptides_gan_pt.py
import torch
import numpy as np
import os
from utils import int_sequence_to_sequence, sequence_to_int_sequence, pad_sequences, PADDING_VALUE, NUM_AMINO_ACIDS # Import utilities
from model import Generator # Import PyTorch Generator
from model import PeptidePredictiveModel # Import PyTorch Predictive Model
from data_loader import load_preprocessed_data # To get max_sequence_length and vocab_size


def sample_from_probs(probabilities, temperature=1.0):
    """
    从概率分布中为每个位置进行采样。
    Args:
        probabilities (np.ndarray): 形状 (max_sequence_length, num_amino_acids_one_hot) 的概率矩阵（NumPy）。
        temperature (float): 控制采样随机性的温度。
    Returns:
        list of int: 采样的整数序列 (1-based indices + padding_value).
    """
    sampled_sequence = []
    num_chars_one_hot = probabilities.shape[-1] # Number of columns in one-hot vector

    for i in range(probabilities.shape[0]): # Iterate over each position
        position_probs = probabilities[i, :]

        # Avoid sampling the padding column if it exists in output probs
        # Note: Our generator outputs probs over NUM_AMINO_ACIDS, which are indices 0 to NUM_AMINO_ACIDS-1,
        # mapping to actual 1-based amino acid indices.
        # So indices 0 to NUM_AMINO_ACIDS-1 from sampling directly map to 1-based amino acid indices 1 to NUM_AMINO_ACIDS.

        # Apply temperature
        position_probs = np.exp(np.log(np.maximum(position_probs, 1e-10)) / temperature)
        position_probs = position_probs / np.sum(position_probs) # Re-normalize

        # Sample amino acid index for the current position (0-based index corresponding to the one-hot column)
        try:
             sampled_index_0based = np.random.choice(num_chars_one_hot, p=position_probs)
        except ValueError:
             print(f"警告: 位置 {i} 的概率分布有问题 ({np.sum(position_probs)}), 使用argmax替代。")
             sampled_index_0based = np.argmax(position_probs)

        # Convert sampled 0-based index back to our 1-based amino acid index or padding value.
        # Since we are generating and training on positive sequences (which shouldn't contain padding or stop char 0 from original data logic),
        # The Generator learns to output probabilities over actual amino acids (indices 1..NUM_AMINO_ACIDS).
        # The One-Hot encoding is over NUM_AMINO_ACIDS columns (mapping to 1..NUM_AMINO_ACIDS).
        # So sampled_index_0based (0 to NUM_AMINO_ACIDS-1) maps directly to 1-based amino acid index (1 to NUM_AMINO_ACIDS) by adding 1.
        sampled_amino_acid_int = sampled_index_0based + 1 # Convert 0-based column index to 1-based amino acid index

        # We are sampling for *all* `max_sequence_length` positions here based on generator output.
        # Handling padding and stop characters like '*' will happen AFTER the full sequence of integers is generated.
        sampled_sequence.append(sampled_amino_acid_int)

    # The generated integer sequence might contain indices corresponding to stop character if present in original alphabet (index 22 if * is last)
    # and WILL contain padding value (0) if the generator is trained to fill remaining positions with padding distribution (unlikely with our current G)
    # We get a fixed-length integer sequence from generator output, then interpret it.
    return sampled_sequence


def generate_and_filter_peptides_gan(
    generator_model_path='models/weights/generator_pt.pth',
    predictive_model_path='models/weights/predictive_model_pt.pth',
    num_peptides_to_generate=1000,
    num_peptides_to_select=10,
    latent_dim=100, # Needs to match generator training
    max_sequence_length=None, # Needs to match GAN and Predictor training
    min_peptide_length=15,
    max_peptide_length=45,
    temperature=0.8 # Controls sampling diversity
):
    """
    使用训练好的GAN生成器生成潜在的抗菌肽序列，并使用预测模型进行筛选（PyTorch）。

    Args:
        generator_model_path (str): 生成器权重路径 (.pth)。
        predictive_model_path (str): 预测模型权重路径 (.pth)。
        num_peptides_to_generate (int): 需要尝试生成的肽链总数。
        num_peptides_to_select (int): 最终选取的（被预测模型认为高概率为抗菌肽的）肽链数量。
        latent_dim (int): 生成器输入的噪声向量维度。
        max_sequence_length (int): 生成器和判别器期望的序列长度。必须提供。
        min_peptide_length (int): 生成肽链的最小长度（非填充非终止符）。
        max_peptide_length (int): 生成肽链的最大长度（非填充非终止符）。
        temperature (float): 控制生成随机性的温度参数。

    Returns:
        tuple: (list of str: selected peptide sequences, list of float: corresponding predictive scores)
    """
    if max_sequence_length is None:
         print("错误：必须提供生成器和判别器训练时使用的最大序列长度 (max_sequence_length)。")
         return [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备进行生成和筛选: {device}")

    num_amino_acids_one_hot = NUM_AMINO_ACIDS # One-Hot dimension
    vocab_size = NUM_AMINO_ACIDS + 1 # Total tokens including padding

    # Load Generator Model
    if os.path.exists(generator_model_path):
        generator = Generator(latent_dim, max_sequence_length, num_amino_acids_one_hot).to(device)
        generator.load_state_dict(torch.load(generator_model_path, map_location=device))
        generator.eval() # Set to evaluation mode
        print(f"成功加载生成器权重来自 {generator_model_path}")
    else:
        print(f"错误：未找到生成器权重文件 {generator_model_path}。请先训练GAN。")
        return [], []

    # Load Predictive Model (for filtering)
    if os.path.exists(predictive_model_path):
        # Predictive model parameters (should match train_predictive_pt.py)
        embedding_dim = 128
        hidden_dim_lstm = 128
        num_filters = 128
        kernel_size = 5
        dropout_rate = 0.3 # Match the dropout used in training, though eval mode disables dropout

        predictive_model = PeptidePredictiveModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim_lstm=hidden_dim_lstm,
            num_filters=num_filters,
            kernel_size=kernel_size,
            max_sequence_length=max_sequence_length,
            dropout_rate=dropout_rate # This parameter is in init, actual dropout layer is handled by model.eval()
        ).to(device)
        predictive_model.load_state_dict(torch.load(predictive_model_path, map_location=device))
        predictive_model.eval() # Set to evaluation mode
        print(f"成功加载预测模型权重来自 {predictive_model_path}")
    else:
        print(f"警告：未找到预测模型权重文件 {predictive_model_path}。将无法对生成的肽链进行筛选。")
        predictive_model = None

    generated_sequences = []
    predicted_scores = []
    generated_count = 0

    print(f"开始生成并筛选潜在的抗菌肽（尝试生成最多 {num_peptides_to_generate} 条，选择 {num_peptides_to_select} 条）...")

    # Generate in batches for efficiency
    batch_size_gen = 64
    num_batches_to_gen = (num_peptides_to_generate + batch_size_gen - 1) // batch_size_gen


    with torch.no_grad(): # Disable gradients for generation and prediction
        for i in range(num_batches_to_gen):
            current_batch_size = min(batch_size_gen, num_peptides_to_generate - len(generated_sequences))
            if current_batch_size <= 0:
                break # Stop if we have enough selected peptides

            noise = torch.randn(current_batch_size, latent_dim, device=device) # Generate noise batch

            # Generator outputs probabilities
            generated_probs_batch = generator(noise).cpu().numpy() # (batch_size_gen, max_len, num_amino_acids_one_hot)

            # Process each generated probability matrix in the batch
            for single_probs in generated_probs_batch: # shape (max_len, num_amino_acids_one_hot)
                 generated_count += 1 # Increment total attempts

                 # Sample discrete integers from probabilities
                 # Sampled integers will be 1-based AA indices, corresponding to one-hot columns 0..NUM_AMINO_ACIDS-1
                 sampled_int_sequence_raw = sample_from_probs(single_probs, temperature) # List of 1-based ints

                 # Convert the potentially full-length sampled integer sequence to string
                 # int_sequence_to_sequence handles stopping at '*' and skipping PADDING_VALUE (0)
                 generated_peptide = int_sequence_to_sequence(sampled_int_sequence_raw, padding_value=PADDING_VALUE)

                 # Filter by desired peptide length (after removing padding and stop)
                 if len(generated_peptide) < min_peptide_length or len(generated_peptide) > max_peptide_length:
                     # print(f"过滤 (长度: {len(generated_peptide)})") # Optional debug
                     continue

                 # print(f"尝试 {generated_count} ({len(generated_sequences)}/ {num_peptides_to_select}): 生成肽链 {generated_peptide} (长度: {len(generated_peptide)})")


                 # Use predictive model for screening
                 prediction_score = -1.0 # Default score if no predictive model

                 if predictive_model is not None:
                     # Convert the generated peptide string back to a padded integer sequence (1-based + padding_value)
                     # The predictor model expects padded 1-based integer sequences.
                     peptide_int_sequence_cleaned = sequence_to_int_sequence(generated_peptide, padding_value=PADDING_VALUE) # Get 1-based int sequence without padding_value char
                     # Pad the cleaned integer sequence for the predictor's fixed input length
                     padded_peptide_for_pred_np = pad_sequences([peptide_int_sequence_cleaned], max_length=max_sequence_length, padding='post', value=PADDING_VALUE) # NumPy array (1, max_len)

                     # Convert to PyTorch Tensor and move to device
                     padded_peptide_for_pred_pt = torch.from_numpy(padded_peptide_for_pred_np).long().to(device) # (1, max_len) LongTensor

                     # Predict probability (predictor outputs logits, apply sigmoid)
                     outputs = predictive_model(padded_peptide_for_pred_pt) # (1, 1) logits
                     prediction_score = torch.sigmoid(outputs).item() # Get the single scalar probability

                     # Filter by prediction score
                     if prediction_score < 0.7: # Adjustable threshold
                          # print(f" -> 过滤 (评分: {prediction_score:.4f})") # Optional debug
                          continue # Skip if score is too low
                     # print(f" -> 保留 (评分: {prediction_score:.4f})") # Optional debug


                 # If we reach here, the peptide is kept
                 generated_sequences.append(generated_peptide)
                 predicted_scores.append(prediction_score)

                 # Stop generating attempts once we have enough selected peptides
                 if len(generated_sequences) >= num_peptides_to_select:
                     print(f"已生成并筛选到 {num_peptides_to_select} 条肽链，停止生成。")
                     break # Break from batch iteration

            if len(generated_sequences) >= num_peptides_to_select:
                 break # Break from batch loop

        # If we still don't have enough after max attempts, use what we have
        if len(generated_sequences) < num_peptides_to_select:
             print(f"尝试生成 {generated_count} 条序列后，未能找到 {num_peptides_to_select} 条符合条件的肽链。找到 {len(generated_sequences)} 条。")


    print(f"\n共尝试生成 {generated_count} 条序列。")
    print(f"经过长度过滤和预测模型筛选后，最终选定 {len(generated_sequences)} 条潜在抗菌肽。")

    # Sort by predictive score (descending) if predictor was used
    if predictive_model is not None:
        sorted_indices = np.argsort(predicted_scores)[::-1] # Descending order
        sorted_sequences = [generated_sequences[i] for i in sorted_indices]
        sorted_scores = [predicted_scores[i] for i in sorted_indices]
    else:
        # If no predictor, just return the ones that passed length filter
        sorted_sequences = generated_sequences
        sorted_scores = predicted_scores


    # Take top N peptides if needed (though loop should have stopped once num_peptides_to_select is reached)
    final_sequences = sorted_sequences[:num_peptides_to_select]
    final_scores = sorted_scores[:num_peptides_to_select]


    print(f"\n选定的 {len(final_sequences)} 条潜在抗菌肽序列（按预测评分降序排列）:")
    for i, peptide in enumerate(final_sequences):
        score_info = f" (评分: {final_scores[i]:.4f})" if predictive_model is not None else ""
        print(f"{i+1}. {peptide}{score_info}")

    return final_sequences, final_scores


if __name__ == '__main__':
    # Determine max_sequence_length from preprocessed data
    data_path = 'data/preprocessed_data.npz'
    determined_max_len = None
    if os.path.exists(data_path):
        try:
            data_info = np.load(data_path, allow_pickle=True)
            if 'max_sequence_length' in data_info:
                 determined_max_len = data_info['max_sequence_length'].item()
                 print(f"从预处理数据获取实际最大序列长度: {determined_max_len}.")
            else:
                 # Fallback to X_train shape if max_sequence_length wasn't explicitly saved
                 if 'X_train' in data_info:
                      determined_max_len = data_info['X_train'].shape[1]
                      print(f"从预处理数据 X_train shape 获取实际最大序列长度: {determined_max_len}.")
                 else:
                      print(f"错误：预处理数据文件 {data_path} 格式不正确或损坏，未找到 'max_sequence_length' 或 'X_train'。")

        except Exception as e:
            print(f"加载预处理数据文件出错: {e}")
            determined_max_len = None
    else:
         print(f"错误：预处理数据文件 {data_path} 不存在。请先运行数据预处理。")

    if determined_max_len is not None:
        # Example generation call
        # Ensure latent_dim matches what was used in GAN training
        latent_dim_used_in_train = 100 # <--- !!! Make sure this matches train_gan_pt.py LATENT_DIM
        selected_peptides, scores = generate_and_filter_peptides_gan(
            max_sequence_length=determined_max_len,
            latent_dim=latent_dim_used_in_train,
            num_peptides_to_generate=2000, # More attempts
            num_peptides_to_select=20, # Select 20 for wet lab
            min_peptide_length=15, # Adjust range based on typical AMP length for wet lab
            max_peptide_length=45,
            temperature=0.9 # Adjust for diversity
        )
    else:
        print("无法进行生成，因为未能确定序列的最大长度。")