import torch
import os
import pickle
import numpy as np
import pandas as pd
from model import Generator, EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM # Import model definitions and hyperparameters
from utils import vocab, MAX_LEN, PAD_TOKEN # Import shared utilities
from data_generated import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder

# Define max length for feature calculation - must be same as used in train_predictive!
MAX_FEATURE_LEN = 20 # Ensure this matches featured_generated.py

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for generation/filtering: {device}")


def calculate_features_for_sequences(sequences):
     """
     Calculates features for a list of sequences using the same functions
     as featured_generated.py. Returns a DataFrame.
     """
     # Handle potentially empty sequences or sequences with unexpected chars
     # Filter out empty/invalid sequences before calculating features
     valid_sequences = [seq for seq in sequences if len(seq) > 0 and all(aa in vocab.word_to_idx for aa in seq)]

     if not valid_sequences:
          print("Warning: No valid sequences to calculate features for.")
          return pd.DataFrame() # Return empty dataframe if no valid sequences

     # Calculate descriptors (call placeholder functions - must be implemented)
     basic_des_df = calculate_basic_descriptors(valid_sequences)
     # autocorrelation_df = calculate_autocorrelation(valid_sequences) # Uncomment when implemented
     # ctd_df = calculate_ctd(valid_sequences) # Uncomment when implemented
     # qso_df = calculate_qso(valid_sequences, max_len=MAX_FEATURE_LEN) # Q-order often needs max_len
     # aac_dpc_tpc_df = calculate_aac_dpc_tpc(valid_sequences) # Uncomment when implemented

     # Combine all features
     # feature_df = pd.concat([basic_des_df, autocorrelation_df, ctd_df, qso_df, aac_dpc_tpc_df], axis=1) # Use all when implemented
     # Using only basic descriptors for now
     feature_df = basic_des_df

     # Ensure column consistency with training data (important for classifier)
     # In a real scenario, you'd load the training feature column names
     # Here, we just return what we got assuming it matches the order/names
     # from featured_generated.py's placeholder implementation.
     # For real use, load original feature_df columns and reindex/align feature_df

     # Add original sequences back (needed for output)
     feature_df['Sequence'] = valid_sequences

     # Handle potential NaNs from feature calculation
     # Simple imputation: fill NaNs with mean from training data features (ideally)
     # Or fill with 0, or a placeholder value. Using mean is common but needs train stats.
     # For now, we'll assume feature calculation doesn't produce NaNs or the classifier handles them.
     # Let's use simple mean imputation for numeric columns:
     numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
     for col in numeric_cols:
         if feature_df[col].isnull().any():
             # Ideally, use the mean from the training set
             # For this example, use the mean of the generated batch (less ideal)
             feature_df[col] = feature_df[col].fillna(feature_df[col].mean())

     return feature_df.drop('Sequence', axis=1), feature_df['Sequence'] # Return features and original sequences

def generate_and_filter_peptides(generator_path, classifier_path, num_to_generate, batch_size_gen, output_dir):
    """
    Loads GAN generator and predictive classifier, generates peptides,
    filters them using the classifier, and saves candidate AMPs.
    """
    # Load Generator
    print(f"Loading Generator model from {generator_path}")
    generator = Generator(vocab_size=vocab.vocab_size,
                          embedding_dim=EMBEDDING_DIM,
                          hidden_dim=HIDDEN_DIM,
                          latent_dim=LATENT_DIM,
                          max_len=MAX_LEN,
                          pad_idx=vocab.pad_idx).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval() # Set to evaluation mode

    # Load Classifier
    print(f"Loading Classifier model from {classifier_path}")
    try:
        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)
        print("Classifier loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Classifier model not found at {classifier_path}. Please train the classifier first.")
        return
    except ImportError:
        print("Error loading classifier: TabPFNClassifier class definition not found. Ensure tabpfn is installed.")
        return
    except Exception as e:
        print(f"An error occurred while loading the classifier: {e}")
        return

    print(f"Generating and filtering {num_to_generate} peptides...")

    generated_candidate_sequences = []

    with torch.no_grad(): # No gradient calculation needed during generation
        for _ in range(0, num_to_generate, batch_size_gen):
            current_batch_size = min(batch_size_gen, num_to_generate - len(generated_candidate_sequences)) # Adjust batch size for last batch
            if current_batch_size <= 0:
                break # Stop if we have enough

            # Generate discrete sequences
            noise = torch.randn(current_batch_size, LATENT_DIM, device=device)
            # Use low temperature or fixed argmax for sampling after training
            generated_indices = generator.generate_discrete(noise, temperature=0.1) # Use low temp

            # Decode sequences and remove padding
            decoded_sequences = [vocab.decode(seq.tolist(), remove_padding=True) for seq in generated_indices]

            # Filter out empty or too-long sequences (optional check based on MAX_LEN)
            decoded_sequences = [seq for seq in decoded_sequences if 0 < len(seq) <= MAX_LEN]

            if not decoded_sequences:
                print(f"Generated {len(decoded_sequences)} valid sequences in this batch. Skipping feature calculation.")
                continue

            # Calculate features for the generated sequences
            # Need to re-run feature calculation for each batch
            # Note: This can be slow for large numbers of generated sequences
            try:
                generated_features, original_generated_sequences = calculate_features_for_sequences(decoded_sequences)
                if generated_features.empty:
                     print("Skipping classification for this batch due to no valid features.")
                     continue
            except Exception as e:
                print(f"Error calculating features for generated sequences: {e}. Skipping batch.")
                continue


            # Use the classifier to predict AMP probability
            try:
                 # Ensure feature columns match the expected order/names by the classifier
                 # This is crucial and might require explicit column handling here
                 # Simple case: assume calculate_features_for_sequences returns columns in the correct order
                 # Better: Load column names from train_predictive's X_train or saved X_test

                 # Dummy column check based on basic descriptor columns
                 # if not all(col.startswith('Basic_') for col in generated_features.columns):
                 #     print("Warning: Generated feature columns might not match expected format. Check descriptor implementations.")

                 predicted_proba = classifier.predict_proba(generated_features)[:, 1] # Probability of class 1 (AMP)

                 # Filter sequences predicted as positive (or above a certain probability threshold)
                 amp_candidates_indices = [i for i, proba in enumerate(predicted_proba) if proba >= 0.5] # Threshold can be adjusted

                 for idx in amp_candidates_indices:
                     generated_candidate_sequences.append(original_generated_sequences.iloc[idx]) # Use iloc to get original string from Series
            except Exception as e:
                print(f"Error during classification filtering: {e}. Skipping batch.")
                # Potentially log the generated sequences that failed classification

            print(f"Generated batch, found {len(amp_candidates_indices)} potential AMP candidates. Total candidates so far: {len(generated_candidate_sequences)}")


    print(f"\nFinished generation and filtering. Found {len(generated_candidate_sequences)} potential AMP candidates.")

    # Save candidate sequences
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join('results/generated_peptides/candidate_amps.csv')

    candidate_df = pd.DataFrame({'Sequence': generated_candidate_sequences})
    candidate_df.to_csv(output_path, index=False)
    print(f"Candidate AMPs saved to {output_path}")


if __name__ == '__main__':
    generator_path = os.path.join('models/generator_model.pth')
    classifier_path = os.path.join('modelspredictive_model.pkl')
    output_dir = 'results/generated_peptides'

    # Adjust parameters for generation
    NUM_GENERATE = 10000 # Number of sequences to attempt to generate
    BATCH_SIZE_GEN = 512 # Batch size for generator inference

    # Ensure descriptor files are callable and return correct format
    # (This script directly imports and calls them)

    generate_and_filter_peptides(generator_path, classifier_path, NUM_GENERATE, BATCH_SIZE_GEN, output_dir)