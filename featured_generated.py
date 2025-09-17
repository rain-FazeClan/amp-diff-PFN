import pandas as pd
import os
import random
from data_generated import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder


def calculate_all_descriptors(sequence, count):
    """
    Calculates all descriptors for a given protein sequence.
    """
    peptides_descriptor = {}
    peptide = str(sequence)

    try:
        # Calculate descriptors for the sequence
        AAC = AAComposition.CalculateAAComposition(peptide)
        DIP = AAComposition.CalculateDipeptideComposition(peptide)
        MBA = Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(peptide, lamba=3)
        CCTD = CTD.CalculateCTD(peptide)
        QSO = QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(peptide, maxlag=3)
        PAAC = PseudoAAC._GetPseudoAAC(peptide, lamda=3)
        APAAC = PseudoAAC.GetAPseudoAAC(peptide, lamda=3)
        Basic = BasicDes.cal_discriptors(peptide)

        # Update the descriptor dictionary
        peptides_descriptor.update(AAC)
        peptides_descriptor.update(DIP)
        peptides_descriptor.update(MBA)
        peptides_descriptor.update(CCTD)
        peptides_descriptor.update(QSO)
        peptides_descriptor.update(PAAC)
        peptides_descriptor.update(APAAC)
        peptides_descriptor.update(Basic)

        if count % 100 == 0:
            print("No.%d  Peptide: %s" % (count, peptide))

    except Exception as e:
        print(f"Error calculating descriptors for peptide {peptide[:10]}...: {e}")

    return peptides_descriptor


def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Assumes script is in root
    grampa_file = os.path.join(base_dir, "origin_data", "grampa.csv")
    negative_file = os.path.join(base_dir, "origin_data", "origin_negative.csv")

    # Ensure the output directory exists
    output_dir = os.path.join(base_dir, "preprocessed_data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "classify.csv")

    # Read and process grampa.csv
    try:
        df_grampa = pd.read_csv(grampa_file)
        print(f"Read {len(df_grampa)} entries from grampa.csv")
        # Extract unique sequences using .unique()
        grampa_sequences = pd.Series(df_grampa["sequence"].unique())
        print(f"{len(grampa_sequences)} unique sequences extracted from grampa.csv")
    except FileNotFoundError:
        print(f"Error: {grampa_file} not found.")
        grampa_sequences = pd.Series(dtype=str)
    except Exception as e:
        print(f"Error processing grampa.csv: {e}")
        grampa_sequences = pd.Series(dtype=str)

    # Read origin_negative.csv
    try:
        df_negative = pd.read_csv(negative_file)
        print(f"Read {len(df_negative)} entries from origin_negative.csv")
        # Extract unique sequences using .unique()
        negative_sequences = pd.Series(df_negative["sequence"].unique())
        print(f"{len(negative_sequences)} unique sequences extracted from origin_negative.csv")
    except FileNotFoundError:
        print(f"Error: {negative_file} not found.")
        negative_sequences = pd.Series(dtype=str)
    except Exception as e:
        print(f"Error processing origin_negative.csv: {e}")
        negative_sequences = pd.Series(dtype=str)

    all_data_with_descriptors = []
    count = 1  # Counter for peptides

    # Process grampa.csv sequences
    print("Processing grampa.csv sequences...")
    for sequence in grampa_sequences:
        if len(sequence) < 6:  # Skip sequences shorter than 6 characters
            continue
        if sequence and isinstance(sequence, str) and all(c.upper() in PseudoAAC.AALetter for c in sequence):
            descriptors = calculate_all_descriptors(sequence.upper(), count)
            descriptors['sequence'] = sequence
            descriptors['label'] = 1
            all_data_with_descriptors.append(descriptors)
            count += 1
        elif sequence:
            print(f"Skipping invalid or non-standard amino acid sequence: {sequence}")
        else:
            print("Skipping empty sequence")

    # Process origin_negative.csv sequences
    print("Processing origin_negative.csv sequences...")
    for sequence in negative_sequences:
        if len(sequence) < 6:  # Skip sequences shorter than 6 characters
            continue
        if sequence and isinstance(sequence, str) and all(c.upper() in PseudoAAC.AALetter for c in sequence):
            descriptors = calculate_all_descriptors(sequence.upper(), count)
            descriptors['sequence'] = sequence
            descriptors['label'] = 0
            all_data_with_descriptors.append(descriptors)
            count += 1
        elif sequence:
            print(f"Skipping invalid or non-standard amino acid sequence: {sequence}")
        else:
            print("Skipping empty sequence")

    # Create DataFrame from all collected data
    if all_data_with_descriptors:
        df_combined = pd.DataFrame(all_data_with_descriptors)

        # Shuffle the data to mix antimicrobial and negative samples
        df_combined = df_combined.sample(frac=1, random_state=random.randint(0, 1000)).reset_index(drop=True)

        # Reorder columns to have 'sequence' and 'label' first
        cols = ['sequence', 'label'] + [col for col in df_combined.columns if col not in ['sequence', 'label']]
        df_combined = df_combined[cols]

        df_combined.to_csv(output_file, index=False)
        print(f"Successfully processed {len(df_combined)} sequences and saved to {output_file}")
    else:
        print("No valid sequences found or processed. Output file will not be created.")


if __name__ == "__main__":
    main()