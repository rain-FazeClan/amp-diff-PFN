import pandas as pd
import os

# Assuming the script is in the root directory and data_generated is a subdirectory
from data_generated import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder


def get_sequence_from_row(row):
    if 'sequence' in row and pd.notna(row['sequence']):
        return str(row['sequence'])
    elif 'Sequence' in row and pd.notna(row['Sequence']):
        return str(row['Sequence'])
    return None


def calculate_all_descriptors(sequence):
    """
    Calculates all descriptors for a given protein sequence.
    """
    descriptors = {}

    # Basic Descriptors
    try:
        basic_des = BasicDes.cal_discriptors(sequence)
        descriptors.update(basic_des)
    except Exception as e:
        print(f"Error calculating BasicDes for sequence {sequence[:10]}...: {e}")

    # AAComposition, DipeptideComposition
    try:
        aac = AAComposition.CalculateAAComposition(sequence)
        descriptors.update(aac)
        dipc = AAComposition.CalculateDipeptideComposition(sequence)
        descriptors.update(dipc)
        # Spectrum (3-mers) was commented out, keeping it that way unless specified
        # spec = AAComposition.GetSpectrumDict(sequence)
        # descriptors.update(spec)
    except Exception as e:
        print(f"Error calculating AAComposition/DipeptideComposition for sequence {sequence[:10]}...: {e}")

    # Autocorrelation descriptors
    try:
        norm_moreau_broto = Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(sequence)
        descriptors.update(norm_moreau_broto)

        moran_auto = Autocorrelation.CalculateMoranAutoTotal(sequence)
        descriptors.update(moran_auto)

        geary_auto = Autocorrelation.CalculateGearyAutoTotal(sequence)
        descriptors.update(geary_auto)
    except Exception as e:
        print(f"Error calculating Autocorrelation for sequence {sequence[:10]}...: {e}")

    # CTD descriptors
    try:
        ctd_descriptors = CTD.CalculateCTD(sequence)
        descriptors.update(ctd_descriptors)
    except Exception as e:
        print(f"Error calculating CTD for sequence {sequence[:10]}...: {e}")

    # PseudoAAC descriptors
    try:
        # Ensure lamda is valid and not greater than sequence length - 1
        lamda_val_pse = 1
        if len(sequence) > 1:
            lamda_val_pse = min(10, len(sequence) - 1)

        pse_aac_type1 = PseudoAAC._GetPseudoAAC(sequence, lamda=lamda_val_pse)
        descriptors.update(pse_aac_type1)

        pse_aac_type2 = PseudoAAC.GetAPseudoAAC(sequence, lamda=lamda_val_pse)
        descriptors.update(pse_aac_type2)
    except Exception as e:
        print(f"Error calculating PseudoAAC for sequence {sequence[:10]}...: {e}")

    # QuasiSequenceOrder descriptors
    try:
        # Ensure maxlag is valid
        max_lag_qso = 1
        if len(sequence) > 1:
            max_lag_qso = min(20, len(sequence) - 1)

        qso_descriptors = QuasiSequenceOrder.GetQuasiSequenceOrder(sequence, maxlag=max_lag_qso)
        descriptors.update(qso_descriptors)
    except Exception as e:
        print(f"Error calculating QuasiSequenceOrder for sequence {sequence[:10]}...: {e}")

    return descriptors


def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Assumes script is in root
    grampa_file = os.path.join(base_dir, "origin_data", "grampa.csv")
    negative_file = os.path.join(base_dir, "origin_data", "origin_negative.csv")

    # Ensure the output directory exists
    output_dir = os.path.join(base_dir, "classify_data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "classify.csv")

    # Read data
    try:
        df_grampa = pd.read_csv(grampa_file)
        print(f"Read {len(df_grampa)} sequences from grampa.csv")
    except FileNotFoundError:
        print(f"Error: {grampa_file} not found.")
        df_grampa = pd.DataFrame()
    except Exception as e:
        print(f"Error reading {grampa_file}: {e}")
        df_grampa = pd.DataFrame()

    try:
        df_negative = pd.read_csv(negative_file)
        print(f"Read {len(df_negative)} sequences from origin_negative.csv")
    except FileNotFoundError:
        print(f"Error: {negative_file} not found.")
        df_negative = pd.DataFrame()
    except Exception as e:
        print(f"Error reading {negative_file}: {e}")
        df_negative = pd.DataFrame()

    all_data_with_descriptors = []
    processed_peptide_count = 0

    # Process grampa.csv (limit to 50 sequences)
    print("Processing grampa.csv...")
    for index, row in df_grampa.iterrows():
        if processed_peptide_count >= 50:
            break
        sequence = get_sequence_from_row(row)
        if sequence and isinstance(sequence, str) and all(c.upper() in PseudoAAC.AALetter for c in sequence):
            processed_peptide_count += 1
            print(f"Processing peptide #{processed_peptide_count} (grampa): {sequence}")

            descriptors = calculate_all_descriptors(sequence.upper())
            descriptors['sequence'] = sequence
            descriptors['label'] = 1
            all_data_with_descriptors.append(descriptors)
        elif sequence:
            print(f"Skipping invalid or non-standard amino acid sequence in grampa.csv row {index + 1}: {sequence}")
        else:
            print(f"Skipping empty sequence in grampa.csv row {index + 1}")

    # Process origin_negative.csv (limit to 50 sequences)
    print("Processing origin_negative.csv...")
    for index, row in df_negative.iterrows():
        if processed_peptide_count >= 50:
            break
        sequence = get_sequence_from_row(row)
        if sequence and isinstance(sequence, str) and all(c.upper() in PseudoAAC.AALetter for c in sequence):
            processed_peptide_count += 1
            print(f"Processing peptide #{processed_peptide_count} (negative): {sequence}")

            descriptors = calculate_all_descriptors(sequence.upper())
            descriptors['sequence'] = sequence
            descriptors['label'] = 0
            all_data_with_descriptors.append(descriptors)
        elif sequence:
            print(
                f"Skipping invalid or non-standard amino acid sequence in origin_negative.csv row {index + 1}: {sequence}")
        else:
            print(f"Skipping empty sequence in origin_negative.csv row {index + 1}")

    # Create DataFrame from all collected data
    if all_data_with_descriptors:
        df_combined = pd.DataFrame(all_data_with_descriptors)

        # Reorder columns to have 'sequence' and 'label' first
        cols = ['sequence', 'label'] + [col for col in df_combined.columns if col not in ['sequence', 'label']]
        df_combined = df_combined[cols]

        df_combined.to_csv(output_file, index=False)
        print(f"Successfully processed {len(df_combined)} sequences and saved to {output_file}")
    else:
        print("No valid sequences found or processed. Output file will not be created.")


if __name__ == "__main__":
    main()