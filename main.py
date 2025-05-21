# main_gan_pt.py
import os
import numpy as np

# Import PyTorch specific modules
from data_loader import preprocess_data, load_preprocessed_data # Use the _pt version
from train_predictive import train_predictive # Use the _pt version
import evaluate_predictive # Use the _pt version
from train import train_gan # Use the _pt version
from generate_peptides import generate_and_filter_peptides_gan # Use the _pt version


# Configuration paths and parameters
GRAMPA_DATA = 'data/grampa.fasta' # Adjust if CSV
NEGATIVE_DATA = 'data/negative_samples.fasta' # Adjust if CSV
PREPROCESSED_DATA_PATH = 'data/preprocessed_data.npz'

PREDICTIVE_WEIGHTS = 'models/weights/predictive_model_pt.pth'
GENERATOR_WEIGHTS = 'models/weights/generator_pt.pth'
DISCRIMINATOR_WEIGHTS = 'models/weights/discriminator_pt.pth'

# Data Preprocessing Parameters
# Set to None to determine automatically from data
# Or set to an integer like 100 if you want a fixed length and potentially truncate longer sequences
MAX_SEQUENCE_LENGTH_PREPROCESS = None # Use None to calculate max length from data

# Predictive Model Training Parameters
PREDICTIVE_EPOCHS = 30
PREDICTIVE_BATCH_SIZE = 64

# GAN Training Parameters
GAN_EPOCHS = 800 # GANs often require many epochs and fine-tuning
GAN_BATCH_SIZE = 128
LATENT_DIM = 100 # Dimension of the noise vector

# Generation and Filtering Parameters
GENERATION_ATTEMPTS = 2000 # Number of samples to try generating from the GAN
GENERATION_COUNT = 20 # Number of potential peptides to select for wet lab
MIN_GENERATED_LENGTH = 15
MAX_GENERATED_LENGTH = 45
GENERATION_TEMPERATURE = 0.8 # Lower values -> less diversity, higher values -> more diversity


if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models/weights', exist_ok=True)

    # 1. Data Preprocessing
    print("===== 数据预处理 (PyTorch流程) =====")
    # preprocess_data saves max_sequence_length and num_amino_acids to the .npz
    actual_max_len_used = preprocess_data(GRAMPA_DATA, NEGATIVE_DATA, PREPROCESSED_DATA_PATH, max_sequence_length=MAX_SEQUENCE_LENGTH_PREPROCESS)

    # Check and get the determined max sequence length from the saved file
    determined_max_len = None
    if os.path.exists(PREPROCESSED_DATA_PATH):
        try:
            data_info = np.load(PREPROCESSED_DATA_PATH, allow_pickle=True)
            if 'max_sequence_length' in data_info:
                 determined_max_len = data_info['max_sequence_length'].item()
                 print(f"从预处理数据文件获取实际最大序列长度: {determined_max_len}")
            else:
                 # Fallback if not explicitly saved (less reliable)
                 if 'X_train' in data_info:
                      determined_max_len = data_info['X_train'].shape[1]
                      print(f"从预处理数据 X_train shape 获取实际最大序列长度: {determined_max_len} (警告: max_sequence_length 元数据缺失)")
                 else:
                      print(f"错误：预处理数据文件 {PREPROCESSED_DATA_PATH} 格式不正确或损坏，无法获取最大序列长度。")

        except Exception as e:
            print(f"加载预处理数据文件出错: {e}")
            determined_max_len = None
    else:
        print("错误：预处理数据文件不存在。请先成功完成数据预处理。")

    if determined_max_len is None:
        print("无法获取序列最大长度，程序退出。")
        exit()

    # 2. Train Predictive Model (Required for filtering generated peptides)
    print("\n===== 训练抗菌肽预测模型 (PyTorch) =====")
    train_predictive(
        data_filepath=PREPROCESSED_DATA_PATH,
        model_save_path=PREDICTIVE_WEIGHTS,
        epochs=PREDICTIVE_EPOCHS,
        batch_size=PREDICTIVE_BATCH_SIZE
    )

    # 3. Evaluate Predictive Model (Optional, but good practice)
    print("\n===== 评估抗菌肽预测模型 (PyTorch) =====")
    evaluate_predictive(
        data_filepath=PREPROCESSED_DATA_PATH,
        model_load_path=PREDICTIVE_WEIGHTS
    )

    # 4. Train GAN
    print("\n===== 训练GAN模型 (PyTorch) =====")
    # The GAN training script will load max_sequence_length internally from data_filepath
    train_gan(
        data_filepath=PREPROCESSED_DATA_PATH, # Uses positive data from this file
        gen_save_path=GENERATOR_WEIGHTS,
        disc_save_path=DISCRIMINATOR_WEIGHTS,
        epochs=GAN_EPOCHS,
        batch_size=GAN_BATCH_SIZE,
        latent_dim=LATENT_DIM
    )

    # 5. Generate and Filter Peptides using GAN
    print("\n===== 使用GAN生成并筛选潜在的抗菌肽 (PyTorch) =====")
    # Need to pass the determined_max_len to the generation function
    generate_and_filter_peptides_gan(
        generator_model_path=GENERATOR_WEIGHTS,
        predictive_model_path=PREDICTIVE_WEIGHTS, # Use the trained predictor
        num_peptides_to_generate=GENERATION_ATTEMPTS,
        num_peptides_to_select=GENERATION_COUNT,
        latent_dim=LATENT_DIM, # Needs to match training LATENT_DIM
        max_sequence_length=determined_max_len, # Pass the determined length
        min_peptide_length=MIN_GENERATED_LENGTH,
        max_peptide_length=MAX_GENERATED_LENGTH,
        temperature=GENERATION_TEMPERATURE
    )

    print("\n===== 所有步骤完成 (PyTorch流程) =====")