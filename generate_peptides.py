import torch
import os
import pickle
import numpy as np
import pandas as pd
from model import DiffusionModel, EMBEDDING_DIM, HIDDEN_DIM, get_diffusion_beta_schedule
from utils import vocab, MAX_LEN, PAD_TOKEN
from featured_generated import calculate_all_descriptors

# Diffusion参数
NUM_DIFFUSION_STEPS = 1000
BETA_SCHEDULE = get_diffusion_beta_schedule(NUM_DIFFUSION_STEPS)
ALPHA = 1. - BETA_SCHEDULE
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)

def sample_ddpm(model, num_steps, shape, device):
    x = torch.randn(shape, device=device)
    for t_ in reversed(range(num_steps)):
        t = torch.full((shape[0],), t_, device=device, dtype=torch.long)
        with torch.no_grad():
            pred_noise = model(x, t)
        a_bar = ALPHA_BAR.to(device)[t_]
        a = ALPHA.to(device)[t_]
        if t_ > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = (1 / torch.sqrt(a)) * (x - (1 - a) / torch.sqrt(1 - a_bar) * pred_noise) + torch.sqrt(BETA_SCHEDULE.to(device)[t_]) * noise
    return x

def onehot_to_token_with_temperature(x, temperature=0.9):
    probs = torch.softmax(x / temperature, dim=-1)
    batch, seq, vocab_size = probs.shape
    probs_2d = probs.view(-1, vocab_size)
    sampled = torch.multinomial(probs_2d, 1).view(batch, seq)
    return sampled

def trim_pad(seq, pad_idx):
    return seq[:seq.index(pad_idx)] if pad_idx in seq else seq

def calculate_features_for_sequences(sequences):
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

def generate_and_filter_peptides(diffusion_model_path, classifier_path, num_to_generate, batch_size_gen, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Diffusion model from {diffusion_model_path}")
    model = DiffusionModel(vocab_size=vocab.vocab_size,
                          embedding_dim=EMBEDDING_DIM,
                          hidden_dim=HIDDEN_DIM,
                          max_len=MAX_LEN,
                          pad_idx=vocab.pad_idx,
                          dropout_rate=0.4).to(device)
    model.load_state_dict(torch.load(diffusion_model_path, map_location=device))
    model.eval()
    print(f"Loading Classifier model from {classifier_path}")
    try:
        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)
        print("Classifier loaded successfully.")
    except Exception as e:
        print(f"Error loading classifier: {e}"); return
    print(f"Generating and filtering {num_to_generate} peptides...")
    generated_candidate_sequences = []
    with torch.no_grad():
        for _ in range(0, num_to_generate, batch_size_gen):
            current_batch_size = min(batch_size_gen, num_to_generate - len(generated_candidate_sequences))
            if current_batch_size <= 0:
                break
            gen_x = sample_ddpm(model, NUM_DIFFUSION_STEPS, (current_batch_size, MAX_LEN, vocab.vocab_size), device)
            gen_tokens = onehot_to_token_with_temperature(gen_x, temperature=0.9)
            decoded_sequences = []
            for seq in gen_tokens.cpu().tolist():
                seq = trim_pad(seq, vocab.pad_idx)
                aa_seq = vocab.decode(seq)
                if 6 <= len(aa_seq) <= MAX_LEN:
                    decoded_sequences.append(aa_seq)
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
                amp_candidates_indices = [i for i, proba in enumerate(predicted_proba) if proba >= 0.5]
                for idx in amp_candidates_indices:
                    generated_candidate_sequences.append(original_generated_sequences.iloc[idx])
            except Exception as e:
                print(f"Error during classification filtering: {e}. Skipping batch.")
            print(f"Generated batch, found {len(amp_candidates_indices)} potential AMP candidates. Total candidates so far: {len(generated_candidate_sequences)}")
    print(f"\nFinished generation and filtering. Found {len(generated_candidate_sequences)} potential AMP candidates.")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'candidate_amps.csv')
    candidate_df = pd.DataFrame({'Sequence': generated_candidate_sequences})
    candidate_df.to_csv(output_path, index=False)
    print(f"Candidate AMPs saved to {output_path}")

if __name__ == '__main__':
    diffusion_model_path = os.path.join('models/diffusion_model_transformer.pth')
    classifier_path = os.path.join('models/predictive_model.pkl')
    output_dir = 'results/generated_peptides'
    NUM_GENERATE = 2000
    BATCH_SIZE_GEN = 256
    generate_and_filter_peptides(diffusion_model_path, classifier_path, NUM_GENERATE, BATCH_SIZE_GEN, output_dir)