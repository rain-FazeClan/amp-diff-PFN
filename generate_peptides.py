import torch
import os
import pickle
import numpy as np
import pandas as pd
import argparse
from model import DiffusionModel, EMBEDDING_DIM, HIDDEN_DIM, get_diffusion_beta_schedule, DROPOUT_RATE
from utils import vocab, DEFAULT_MAX_LEN, PAD_TOKEN
from featured_generated import calculate_all_descriptors


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Generate AMP peptides using trained diffusion model')

    # 模型路径参数
    parser.add_argument('--diffusion_model', type=str,
                        default='models/diffusion_model_transformer.pth',
                        help='Path to trained diffusion model (default: models/diffusion_model_transformer.pth)')
    parser.add_argument('--classifier_model', type=str,
                        default='models/predictive_model.pkl',
                        help='Path to trained classifier model (default: models/predictive_model.pkl)')

    # 生成参数
    parser.add_argument('--num_generate', type=int, default=4000,
                        help='Total number of peptides to generate (default: 4000)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for generation (default: 256)')
    parser.add_argument('--max_len', type=int, default=DEFAULT_MAX_LEN,
                        help=f'Maximum sequence length (default: {DEFAULT_MAX_LEN})')

    # 采样参数
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='Sampling temperature for diversity (default: 1.5)')
    parser.add_argument('--min_length', type=int, default=6,
                        help='Minimum valid sequence length (default: 6)')

    # 过滤参数
    parser.add_argument('--amp_threshold', type=float, default=0.8,
                        help='AMP classification threshold (default: 0.8)')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results/generated_peptides',
                        help='Output directory for results (default: results/generated_peptides)')
    parser.add_argument('--output_file', type=str, default='candidate_amps.csv',
                        help='Output filename (default: candidate_amps.csv)')

    # 其他参数
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


# Diffusion参数 - 与训练保持一致
NUM_DIFFUSION_STEPS = 1000
BETA_SCHEDULE = get_diffusion_beta_schedule(NUM_DIFFUSION_STEPS, beta_start=1e-3, beta_end=0.05)  # 更新噪声调度
ALPHA = 1. - BETA_SCHEDULE
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)

def sample_ddpm(model, num_steps, shape, device, alpha, alpha_bar, beta_schedule):
    """更新的DDPM采样函数，与训练脚本保持一致"""
    model.eval()
    x = torch.randn(shape, device=device)

    with torch.no_grad():
        for t_ in reversed(range(num_steps)):
            t = torch.full((shape[0],), t_, device=device, dtype=torch.long)
            pred_noise = model(x, t)
            a_bar = alpha_bar.to(device)[t_]
            a = alpha.to(device)[t_]

            if t_ > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(a)) * (x - (1 - a) / torch.sqrt(1 - a_bar) * pred_noise) + torch.sqrt(beta_schedule.to(device)[t_]) * noise

    model.train()
    return x

def onehot_to_token_with_temperature(x, temperature=1.0):
    """更新温度采样函数，默认温度调整为1.0"""
    probs = torch.softmax(x / temperature, dim=-1)
    batch, seq, vocab_size = probs.shape
    probs_2d = probs.view(-1, vocab_size)
    sampled = torch.multinomial(probs_2d, 1).view(batch, seq)
    return sampled

def trim_pad(seq, pad_idx):
    return seq[:seq.index(pad_idx)] if pad_idx in seq else seq

def calculate_features_for_sequences(sequences):
    """计算序列特征的函数"""
    valid_sequences = [seq for seq in sequences if len(seq) > 0 and all(aa in vocab.word_to_idx for aa in seq)]
    if not valid_sequences:
        print("Warning: No valid sequences to calculate features for.")
        return pd.DataFrame(), []

    count = 1
    descriptors_list = []
    for seq in valid_sequences:
        descriptors = calculate_all_descriptors(seq, count)
        descriptors_list.append(descriptors)
        count += 1

    feature_df = pd.DataFrame(descriptors_list)
    feature_df['Sequence'] = valid_sequences
    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        if feature_df[col].isnull().any():
            feature_df[col] = feature_df[col].fillna(feature_df[col].mean())

    return feature_df.drop('Sequence', axis=1), feature_df['Sequence']

def generate_and_filter_peptides(diffusion_model_path, classifier_path, num_to_generate, batch_size_gen,
                               output_dir, max_len=DEFAULT_MAX_LEN, temperature=2.0, min_length=6,
                               amp_threshold=0.5, output_file='candidate_amps.csv', verbose=False):
    """主要的生成和过滤函数，更新模型初始化参数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Device: {device}")
        print(f"Parameters:")
        print(f"  - Model paths: {diffusion_model_path}, {classifier_path}")
        print(f"  - Generation: {num_to_generate} peptides, batch_size={batch_size_gen}")
        print(f"  - Sequence: max_len={max_len}, min_len={min_length}")
        print(f"  - Sampling: temperature={temperature}")
        print(f"  - Filtering: AMP threshold={amp_threshold}")
        print(f"  - Output: {output_dir}/{output_file}")
        print("-" * 50)

    print(f"Loading Diffusion model from {diffusion_model_path}")

    # 更新模型初始化，使用训练时的dropout参数
    model = DiffusionModel(vocab_size=vocab.vocab_size,
                          embedding_dim=EMBEDDING_DIM,
                          hidden_dim=HIDDEN_DIM,
                          max_len=max_len,
                          pad_idx=vocab.pad_idx,
                          dropout_rate=DROPOUT_RATE).to(device)  # 使用训练时的dropout rate

    model.load_state_dict(torch.load(diffusion_model_path, map_location=device))
    model.eval()

    print(f"Loading Classifier model from {classifier_path}")
    try:
        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)
        print("Classifier loaded successfully.")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return

    print(f"Generating and filtering {num_to_generate} peptides...")
    generated_candidate_sequences = []
    total_generated = 0
    total_valid = 0

    with torch.no_grad():
        batch_count = 0
        for _ in range(0, num_to_generate, batch_size_gen):
            current_batch_size = min(batch_size_gen, num_to_generate - len(generated_candidate_sequences))
            if current_batch_size <= 0:
                break

            batch_count += 1
            if verbose:
                print(f"\nBatch {batch_count}: Generating {current_batch_size} sequences...")

            # 使用更新的采样函数
            gen_x = sample_ddpm(model, NUM_DIFFUSION_STEPS,
                               (current_batch_size, max_len, vocab.vocab_size),
                               device, ALPHA, ALPHA_BAR, BETA_SCHEDULE)

            # 使用指定的温度来增加多样性
            gen_tokens = onehot_to_token_with_temperature(gen_x, temperature=temperature)

            decoded_sequences = []
            for seq in gen_tokens.cpu().tolist():
                seq = trim_pad(seq, vocab.pad_idx)
                aa_seq = vocab.decode(seq)
                total_generated += 1
                if min_length <= len(aa_seq) <= max_len:  # 序列长度过滤
                    decoded_sequences.append(aa_seq)
                    total_valid += 1

            if verbose:
                print(f"  Valid sequences: {len(decoded_sequences)}/{current_batch_size}")

            if not decoded_sequences:
                print(f"Generated {len(decoded_sequences)} valid sequences in this batch. Skipping feature calculation.")
                continue

            try:
                generated_features, original_generated_sequences = calculate_features_for_sequences(decoded_sequences)
                if generated_features.empty:
                    print("Skipping classification for this batch due to no valid features.")
                    continue
            except Exception as e:
                print(f"Error calculating features for generated sequences: {e}. Skipping batch.")
                continue

            try:
                predicted_proba = classifier.predict_proba(generated_features)[:, 1]
                amp_candidates_indices = [i for i, proba in enumerate(predicted_proba) if proba >= amp_threshold]
                for idx in amp_candidates_indices:
                    generated_candidate_sequences.append(original_generated_sequences.iloc[idx])

                if verbose:
                    print(f"  AMP candidates: {len(amp_candidates_indices)}/{len(decoded_sequences)}")
                    if len(amp_candidates_indices) > 0:
                        avg_prob = np.mean([predicted_proba[i] for i in amp_candidates_indices])
                        print(f"  Average AMP probability: {avg_prob:.3f}")

            except Exception as e:
                print(f"Error during classification filtering: {e}. Skipping batch.")

            if not verbose:
                print(f"Generated batch {batch_count}, found {len(amp_candidates_indices) if 'amp_candidates_indices' in locals() else 0} potential AMP candidates. Total candidates so far: {len(generated_candidate_sequences)}")

    print(f"\nFinished generation and filtering. Found {len(generated_candidate_sequences)} potential AMP candidates.")
    print(f"Generation efficiency: {total_valid}/{total_generated} valid sequences ({100*total_valid/total_generated:.1f}%)")
    print(f"AMP discovery rate: {len(generated_candidate_sequences)}/{total_valid} candidates ({100*len(generated_candidate_sequences)/total_valid:.1f}%)")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    candidate_df = pd.DataFrame({'Sequence': generated_candidate_sequences})
    candidate_df.to_csv(output_path, index=False)
    print(f"Candidate AMPs saved to {output_path}")

    # 计算并显示生成统计
    if len(generated_candidate_sequences) > 0:
        avg_length = sum(len(seq) for seq in generated_candidate_sequences) / len(generated_candidate_sequences)
        unique_sequences = len(set(generated_candidate_sequences))
        diversity_score = unique_sequences / len(generated_candidate_sequences)

        # 序列长度分布
        lengths = [len(seq) for seq in generated_candidate_sequences]
        length_distribution = {}
        for length in set(lengths):
            length_distribution[length] = lengths.count(length)

        print(f"\nGeneration Statistics:")
        print(f"Average sequence length: {avg_length:.1f}")
        print(f"Unique sequences: {unique_sequences}/{len(generated_candidate_sequences)}")
        print(f"Diversity score: {diversity_score:.4f}")
        print(f"Length range: {min(lengths)}-{max(lengths)}")

        if verbose:
            print(f"Length distribution: {dict(sorted(length_distribution.items()))}")

        # 保存详细统计信息
        stats_path = os.path.join(output_dir, 'generation_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Generation Statistics\n")
            f.write(f"=====================\n")
            f.write(f"Total generated: {total_generated}\n")
            f.write(f"Valid sequences: {total_valid} ({100*total_valid/total_generated:.1f}%)\n")
            f.write(f"AMP candidates: {len(generated_candidate_sequences)} ({100*len(generated_candidate_sequences)/total_valid:.1f}%)\n")
            f.write(f"Average length: {avg_length:.1f}\n")
            f.write(f"Unique sequences: {unique_sequences}/{len(generated_candidate_sequences)}\n")
            f.write(f"Diversity score: {diversity_score:.4f}\n")
            f.write(f"Length range: {min(lengths)}-{max(lengths)}\n")
            f.write(f"Length distribution: {dict(sorted(length_distribution.items()))}\n")
            f.write(f"\nParameters used:\n")
            f.write(f"Temperature: {temperature}\n")
            f.write(f"AMP threshold: {amp_threshold}\n")
            f.write(f"Min/Max length: {min_length}/{max_len}\n")
        print(f"Detailed statistics saved to {stats_path}")

if __name__ == '__main__':
    args = parse_args()

    # 设置随机种子以便复现结果
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    generate_and_filter_peptides(
        diffusion_model_path=args.diffusion_model,
        classifier_path=args.classifier_model,
        num_to_generate=args.num_generate,
        batch_size_gen=args.batch_size,
        output_dir=args.output_dir,
        max_len=args.max_len,
        temperature=args.temperature,
        min_length=args.min_length,
        amp_threshold=args.amp_threshold,
        output_file=args.output_file,
        verbose=args.verbose
    )
