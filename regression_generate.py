import pandas as pd
import os
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    grampa_file = os.path.join(base_dir, "origin_data", "grampa.csv")

    # Ensure the output directory exists
    output_dir = os.path.join(base_dir, "preprocessed_data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "regression.csv")

    # Read and process grampa.csv
    try:
        df_grampa = pd.read_csv(grampa_file)
        print(f"Read {len(df_grampa)} entries from grampa.csv")
        
        # Group by sequence and find minimum value (MIC)
        df_grampa_min = df_grampa.groupby('sequence')['value'].min().reset_index()
        print(f"Found {len(df_grampa_min)} unique sequences with minimum values")
        
    except FileNotFoundError:
        print(f"Error: {grampa_file} not found.")
        return
    except Exception as e:
        print(f"Error processing grampa.csv: {e}")
        return

    all_data_with_descriptors = []
    count = 1

    # Process grampa.csv sequences
    print("Processing grampa.csv sequences...")
    for _, row in df_grampa_min.iterrows():
        sequence = row['sequence']
        mic_value = row['value']
        
        if len(sequence) < 6:  # Skip sequences shorter than 6 characters
            continue
        if sequence and isinstance(sequence, str) and all(c.upper() in PseudoAAC.AALetter for c in sequence):
            descriptors = calculate_all_descriptors(sequence.upper(), count)
            descriptors['sequence'] = sequence
            descriptors['MIC'] = mic_value  # Use MIC instead of label
            all_data_with_descriptors.append(descriptors)
            count += 1
        elif sequence:
            print(f"Skipping invalid or non-standard amino acid sequence: {sequence}")
        else:
            print("Skipping empty sequence")

    # Create DataFrame from all collected data
    if all_data_with_descriptors:
        df_combined = pd.DataFrame(all_data_with_descriptors)

        # Reorder columns to have 'sequence' and 'MIC' first
        cols = ['sequence', 'MIC'] + [col for col in df_combined.columns if col not in ['sequence', 'MIC']]
        df_combined = df_combined[cols]

        df_combined.to_csv(output_file, index=False)
        print(f"Successfully processed {len(df_combined)} sequences and saved to {output_file}")
    else:
        print("No valid sequences found or processed. Output file will not be created.")


if __name__ == "__main__":
    main()