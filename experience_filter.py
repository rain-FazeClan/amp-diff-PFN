import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os
import argparse

class ExperienceFilter:
    def __init__(self):
        self.hydrophobic_aa = set(['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'])
        self.hydrophilic_aa = set(['R', 'K', 'D', 'E', 'Q', 'N', 'H', 'S', 'T', 'G', 'P', 'C'])
        self.positively_charged_aa = set(['R', 'K', 'H'])
        self.negatively_charged_aa = set(['D', 'E'])

    def calculate_amphipathicity(self, sequence):
        hydrophobic_count = sum(1 for aa in sequence if aa in self.hydrophobic_aa)
        hydrophilic_count = sum(1 for aa in sequence if aa in self.hydrophilic_aa)
        total_count = len(sequence)
        if total_count == 0:
            return 0
        hydrophobic_ratio = hydrophobic_count / total_count
        hydrophilic_ratio = hydrophilic_count / total_count
        if hydrophobic_ratio + hydrophilic_ratio == 0:
            return 0
        amphipathicity = 2 * hydrophobic_ratio * hydrophilic_ratio / (hydrophobic_ratio + hydrophilic_ratio)
        return amphipathicity

    def is_amphipathic(self, sequence, min_amphipathicity=0.35):
        hydrophobic_count = sum(1 for aa in sequence if aa in self.hydrophobic_aa)
        hydrophilic_count = sum(1 for aa in sequence if aa in self.hydrophilic_aa)
        if hydrophobic_count == 0 or hydrophilic_count == 0:
            return False
        amphipathicity = self.calculate_amphipathicity(sequence)
        return amphipathicity >= min_amphipathicity

    def check_length(self, sequence, min_length=15, max_length=18):
        return min_length <= len(sequence) <= max_length

    def calculate_net_charge_at_ph(self, sequence, ph=7.4):
        try:
            protein_analysis = ProteinAnalysis(sequence)
            net_charge = protein_analysis.charge_at_pH(ph)
            return net_charge
        except Exception as e:
            print(f"Error calculating charge for sequence {sequence}: {e}")
            return 0

    def is_positively_charged(self, sequence, ph=7.4, min_charge=3.5):
        net_charge = self.calculate_net_charge_at_ph(sequence, ph)
        return net_charge >= min_charge

    def calculate_additional_properties(self, sequence):
        try:
            protein_analysis = ProteinAnalysis(sequence)
            properties = {
                'molecular_weight': protein_analysis.molecular_weight(),
                'aromaticity': protein_analysis.aromaticity(),
                'instability_index': protein_analysis.instability_index(),
                'isoelectric_point': protein_analysis.isoelectric_point(),
                'gravy': protein_analysis.gravy(),
                'helix_fraction': protein_analysis.secondary_structure_fraction()[0],
                'turn_fraction': protein_analysis.secondary_structure_fraction()[1],
                'sheet_fraction': protein_analysis.secondary_structure_fraction()[2]
            }
            return properties
        except Exception as e:
            print(f"Error calculating properties for sequence {sequence}: {e}")
            return {}

    def filter_peptides(self, input_file, output_file=None,
                       min_amphipathicity=0.35,
                       min_length=15, max_length=18,
                       ph=7.4, min_charge=3.5):
        if not os.path.exists(input_file):
            print(f"输入文件 {input_file} 不存在")
            return None

        df = pd.read_csv(input_file)

        if 'Sequence' not in df.columns:
            print("输入文件中没有找到序列列，请确保列名为'Sequence'")
            return None

        print(f"开始筛选，共有 {len(df)} 个候选肽")
        print("筛选条件:")
        print(f"  序列长度: {min_length}-{max_length}")
        print(f"  两亲性指数: ≥{min_amphipathicity}")
        print(f"  净电荷 (pH {ph}): ≥{min_charge}")

        filtered_results = []

        for idx, row in df.iterrows():
            sequence = str(row['Sequence']).strip().upper()
            if not sequence or not sequence.isalpha():
                continue

            length_pass = self.check_length(sequence, min_length, max_length)
            amphipathic_pass = self.is_amphipathic(sequence, min_amphipathicity)
            charge_pass = self.is_positively_charged(sequence, ph, min_charge)

            if length_pass and amphipathic_pass and charge_pass:
                properties = self.calculate_additional_properties(sequence)
                result = {
                    'Sequence': sequence,
                    'length': len(sequence),
                    'amphipathicity': self.calculate_amphipathicity(sequence),
                    'net_charge_ph7.4': self.calculate_net_charge_at_ph(sequence, ph),
                    'passes_length': length_pass,
                    'passes_amphipathicity': amphipathic_pass,
                    'passes_charge': charge_pass,
                    'overall_pass': True
                }
                for col in df.columns:
                    if col != 'Sequence' and col not in result:
                        result[col] = row[col]
                result.update(properties)
                filtered_results.append(result)

        if filtered_results:
            filtered_df = pd.DataFrame(filtered_results)
            print(f"筛选完成，通过筛选的肽段数量: {len(filtered_df)}")
            print(f"筛选通过率: {len(filtered_df)/len(df)*100:.2f}%")
            print("\n筛选结果统计:")
            print(f"长度范围: {filtered_df['length'].min()}-{filtered_df['length'].max()}")
            print(f"两亲性指数范围: {filtered_df['amphipathicity'].min():.3f}-{filtered_df['amphipathicity'].max():.3f}")
            print(f"净电荷范围: {filtered_df['net_charge_ph7.4'].min():.3f}-{filtered_df['net_charge_ph7.4'].max():.3f}")
            if output_file:
                filtered_df.to_csv(output_file, index=False)
                print(f"筛选结果已保存到: {output_file}")
            return filtered_df
        else:
            print("没有肽通过所有筛选条件")
            return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='对候选AMP进行经验筛选')
    parser.add_argument('--input', help='输入的候选肽CSV文件路径',
                        default='results/generated_peptides/candidate_amps.csv')
    parser.add_argument('--output', help='输出文件路径',
                       default='results/generated_peptides/potential_amps.csv')
    parser.add_argument('--min_amphipathicity', type=float, default=0.35,
                       help='最小两亲性指数 (默认: 0.35)')
    parser.add_argument('--min_length', type=int, default=15,
                       help='最小序列长度 (默认: 15)')
    parser.add_argument('--max_length', type=int, default=18,
                       help='最大序列长度 (默认: 18)')
    parser.add_argument('--ph', type=float, default=7.4,
                       help='计算净电荷的pH值 (默认: 7.4)')
    parser.add_argument('--min_charge', type=float, default=3.5,
                       help='最小净正电荷 (默认: 3.5)')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    filter_obj = ExperienceFilter()
    filtered_df = filter_obj.filter_peptides(
        input_file=args.input,
        output_file=args.output,
        min_amphipathicity=args.min_amphipathicity,
        min_length=args.min_length,
        max_length=args.max_length,
        ph=args.ph,
        min_charge=args.min_charge
    )
    if filtered_df is not None and len(filtered_df) > 0:
        print(f"\n筛选成功完成！共筛选出 {len(filtered_df)} 个符合条件的AMP候选肽")
    else:
        print("筛选未发现符合条件的肽")

if __name__ == "__main__":
    main()