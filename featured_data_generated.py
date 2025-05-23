import pandas as pd
import os
# Import descriptor calculation functions from data_generated
from data_generated import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder

DATA_DIR = 'data'
CLASSIFY_DATA_DIR = 'classify_data/'
GRAMPA_FILE = 'origin_data/grampa.csv'
NEGATIVE_FILE = 'origin_data/origin_negative.csv'
OUTPUT_FILE = 'classify_data.csv'

# Define max length for feature calculation, important for Q-order and some others
# Should align with MAX_LEN for GAN or be larger if needed for features
MAX_FEATURE_LEN = 20

def generate_features(grampa_path, negative_path, output_dir):
    """
    Loads peptide sequences, calculates features, labels them, and saves to CSV.
    """
    # Load data
    grampa_df = pd.read_csv(grampa_path)
    negative_df = pd.read_csv(negative_path)

    # Add label column
    grampa_df['label'] = 1
    negative_df['label'] = 0

    # Combine
    combined_df = pd.concat([grampa_df, negative_df], ignore_index=True)

    # Filter sequences by length if necessary for features
    # combined_df = combined_df[combined_df['Sequence'].str.len() <= MAX_FEATURE_LEN] # Optional, depending on descriptor tool limits

    sequences = combined_df['Sequence'].tolist()
    labels = combined_df['label'].tolist()

    print(f"Processing {len(sequences)} sequences for feature generation...")

    # Calculate descriptors (call placeholder functions)
    basic_des_df = calculate_basic_descriptors(sequences)
    # autocorrelation_df = calculate_autocorrelation(sequences) # Uncomment when implemented
    # ctd_df = calculate_ctd(sequences) # Uncomment when implemented
    # qso_df = calculate_qso(sequences, max_len=MAX_FEATURE_LEN) # Q-order often needs max_len
    # aac_dpc_tpc_df = calculate_aac_dpc_tpc(sequences) # Uncomment when implemented

    # Combine all features
    # feature_df = pd.concat([basic_des_df, autocorrelation_df, ctd_df, qso_df, aac_dpc_tpc_df], axis=1) # Use all when implemented
    # Using only basic descriptors for now
    feature_df = basic_des_df

    # Add original sequences and labels back
    feature_df['Sequence'] = sequences
    feature_df['label'] = labels

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, OUTPUT_FILE)

    # Save
    feature_df.to_csv(output_path, index=False)
    print(f"Feature generation complete. Saved to {output_path}")
    print(f"Output shape: {feature_df.shape}")
    print(f"Columns: {feature_df.columns.tolist()}")


if __name__ == '__main__':
    grampa_path = os.path.join(DATA_DIR, GRAMPA_FILE)
    negative_path = os.path.join(DATA_DIR, NEGATIVE_FILE)
    generate_features(grampa_path, negative_path, CLASSIFY_DATA_DIR)