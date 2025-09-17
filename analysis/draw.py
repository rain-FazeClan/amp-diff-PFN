import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Tuple
import re

# Set font and plot style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def plot_amp_length_distribution(csv_path):
    """Plot AMP length distribution"""
    df = pd.read_csv(csv_path)
    df_pos = df[df['label'] == 1]

    if 'sequence' not in df_pos.columns:
        raise ValueError("CSV file missing 'sequence' column")

    lengths = df_pos['sequence'].apply(len)
    lengths = lengths[(lengths >= 6) & (lengths <= 75)]
    length_counts = lengths.value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    plt.bar(length_counts.index, length_counts.values, color='skyblue', alpha=0.7, edgecolor='navy')
    plt.xlabel('Peptide Length (amino acids)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Positive Sample AMP Length Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def parse_training_log_from_text(log_content: str) -> Dict:
    """Parse key metrics from log text content"""
    # Extract training data
    epochs = []
    losses = []
    learning_rates = []
    patience_counts = []

    # Reconstruction accuracy data
    recon_data = {
        'epoch': [],
        't20': [], 't50': [], 't100': [], 't200': []
    }

    # Diversity data
    diversity_data = {
        'epoch': [],
        'valid_sequences': [],
        'total_sequences': [],
        'diversity_score': [],
        'avg_length': []
    }

    lines = log_content.split('\n')
    current_epoch = None

    for line in lines:
        # Extract epoch, loss, learning rate
        epoch_match = re.search(r'Epoch (\d+) finished\. Avg Loss: ([\d.]+), LR: ([\d.e-]+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs.append(current_epoch)
            losses.append(float(epoch_match.group(2)))
            learning_rates.append(float(epoch_match.group(3)))

        # Extract patience count
        patience_match = re.search(r'Patience: (\d+)/(\d+)', line)
        if patience_match and current_epoch is not None:
            patience_counts.append((current_epoch, int(patience_match.group(1))))

        # Extract reconstruction accuracy
        if 'Reconstruction Accuracy:' in line and current_epoch is not None:
            recon_data['epoch'].append(current_epoch)

        recon_match = re.search(r't=(\d+): ([\d.]+)', line)
        if recon_match and len(recon_data['epoch']) > 0 and current_epoch == recon_data['epoch'][-1]:
            t_val = recon_match.group(1)
            acc_val = float(recon_match.group(2))
            if t_val in ['20', '50', '100', '200']:
                # Ensure each timestep has corresponding data
                while len(recon_data[f't{t_val}']) < len(recon_data['epoch']):
                    recon_data[f't{t_val}'].append(acc_val)

        # Extract diversity data
        diversity_match = re.search(r'Valid sequences: (\d+)/(\d+)', line)
        if diversity_match and current_epoch is not None:
            valid_seq = int(diversity_match.group(1))
            total_seq = int(diversity_match.group(2))
            diversity_data['epoch'].append(current_epoch)
            diversity_data['valid_sequences'].append(valid_seq)
            diversity_data['total_sequences'].append(total_seq)

        diversity_score_match = re.search(r'Generation diversity: ([\d.]+), Avg length: ([\d.]+)', line)
        if diversity_score_match and len(diversity_data['diversity_score']) < len(diversity_data['epoch']):
            diversity_data['diversity_score'].append(float(diversity_score_match.group(1)))
            diversity_data['avg_length'].append(float(diversity_score_match.group(2)))

    # Ensure reconstruction data consistency
    for t_step in ['20', '50', '100', '200']:
        while len(recon_data[f't{t_step}']) < len(recon_data['epoch']):
            recon_data[f't{t_step}'].append(0.0)  # Fill missing values

    return {
        'epochs': epochs,
        'losses': losses,
        'learning_rates': learning_rates,
        'patience_counts': patience_counts,
        'reconstruction': recon_data,
        'diversity': diversity_data
    }


def parse_training_log(log_file_path: str) -> Dict:
    """Parse training log file and extract key metrics"""
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return parse_training_log_from_text(content)


def plot_training_metrics(log_data: Dict, save_path: str = None):
    """Plot comprehensive training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Diffusion Model Training Analysis', fontsize=16, fontweight='bold')

    # 1. Loss curve
    ax1 = axes[0, 0]
    ax1.plot(log_data['epochs'], log_data['losses'], 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)

    # Add best loss point
    min_loss_idx = np.argmin(log_data['losses'])
    ax1.scatter(log_data['epochs'][min_loss_idx], log_data['losses'][min_loss_idx],
                color='red', s=100, zorder=5, label=f'Best Loss: {log_data["losses"][min_loss_idx]:.4f}')
    ax1.legend()

    # 2. Learning rate schedule
    ax2 = axes[0, 1]
    ax2.semilogy(log_data['epochs'], log_data['learning_rates'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate (log scale)')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)

    # 3. Reconstruction accuracy by timestep
    ax3 = axes[0, 2]
    recon_data = log_data['reconstruction']
    if recon_data['epoch']:
        for t_step in ['20', '50', '100', '200']:
            if recon_data[f't{t_step}']:
                ax3.plot(recon_data['epoch'], recon_data[f't{t_step}'],
                        marker='o', label=f't={t_step}', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Reconstruction Accuracy')
    ax3.set_title('Reconstruction Accuracy by Timestep')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Generation diversity
    ax4 = axes[1, 0]
    div_data = log_data['diversity']
    if div_data['epoch']:
        ax4.plot(div_data['epoch'], div_data['diversity_score'], 'purple',
                marker='s', linewidth=2, markersize=6, label='Diversity Score')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Diversity Score')
        ax4.set_title('Generation Diversity')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)
        ax4.legend()

    # 5. Valid sequence generation rate
    ax5 = axes[1, 1]
    if div_data['epoch']:
        valid_ratios = [v/t for v, t in zip(div_data['valid_sequences'], div_data['total_sequences'])]
        ax5.plot(div_data['epoch'], valid_ratios, 'orange',
                marker='^', linewidth=2, markersize=6, label='Valid Sequence Rate')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Valid Sequence Ratio')
        ax5.set_title('Valid Sequence Generation Rate')
        ax5.set_ylim(0, 1.1)
        ax5.grid(True, alpha=0.3)
        ax5.legend()

    # 6. Average sequence length
    ax6 = axes[1, 2]
    if div_data['epoch']:
        ax6.plot(div_data['epoch'], div_data['avg_length'], 'brown',
                marker='d', linewidth=2, markersize=6, label='Average Length')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Average Sequence Length')
        ax6.set_title('Generated Sequence Average Length')
        ax6.grid(True, alpha=0.3)
        ax6.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_reconstruction_heatmap(log_data: Dict, save_path: str = None):
    """Plot reconstruction accuracy heatmap"""
    recon_data = log_data['reconstruction']

    if not recon_data['epoch']:
        print("No reconstruction accuracy data")
        return

    # Prepare heatmap data
    epochs = recon_data['epoch']
    timesteps = ['t=20', 't=50', 't=100', 't=200']

    # Create data matrix
    data_matrix = []
    for t_step in ['20', '50', '100', '200']:
        data_matrix.append(recon_data[f't{t_step}'])

    data_matrix = np.array(data_matrix)

    plt.figure(figsize=(12, 6))
    sns.heatmap(data_matrix,
                xticklabels=epochs,
                yticklabels=timesteps,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Reconstruction Accuracy'})

    plt.title('Reconstruction Accuracy Heatmap by Timestep', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Diffusion Timestep', fontsize=12)
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")

    plt.tight_layout()
    plt.show()


def plot_early_stopping_analysis(log_data: Dict, save_path: str = None):
    """Plot early stopping mechanism analysis"""
    patience_data = log_data['patience_counts']

    if not patience_data:
        print("No early stopping data")
        return

    epochs_with_patience = [item[0] for item in patience_data]
    patience_values = [item[1] for item in patience_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: Loss curve with early stopping markers
    ax1.plot(log_data['epochs'], log_data['losses'], 'b-', linewidth=2, alpha=0.7, label='Training Loss')

    # Mark early stopping patience points
    for epoch, patience in patience_data:
        if patience > 0:  # Only show points with patience count
            loss_idx = log_data['epochs'].index(epoch) if epoch in log_data['epochs'] else None
            if loss_idx is not None:
                ax1.scatter(epoch, log_data['losses'][loss_idx],
                           color='red', s=50, alpha=0.6, zorder=5)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve with Early Stopping Markers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Patience count changes
    ax2.plot(epochs_with_patience, patience_values, 'ro-', linewidth=2, markersize=6)
    ax2.fill_between(epochs_with_patience, patience_values, alpha=0.3, color='red')
    ax2.axhline(y=30, color='orange', linestyle='--', linewidth=2, label='Early Stop Threshold (30)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Patience Count')
    ax2.set_title('Early Stopping Patience Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Early stopping analysis saved to: {save_path}")

    plt.show()


def generate_training_summary_report(log_data: Dict, save_path: str = None):
    """Generate training summary report"""
    report = []
    report.append("=" * 60)
    report.append("Diffusion Model Training Summary Report")
    report.append("=" * 60)

    # Basic training information
    total_epochs = len(log_data['epochs'])
    final_loss = log_data['losses'][-1]
    best_loss = min(log_data['losses'])
    best_epoch = log_data['epochs'][log_data['losses'].index(best_loss)]

    report.append(f"\n[Basic Training Information]")
    report.append(f"Total training epochs: {total_epochs}")
    report.append(f"Final loss: {final_loss:.4f}")
    report.append(f"Best loss: {best_loss:.4f} (epoch {best_epoch})")
    report.append(f"Loss improvement: {((log_data['losses'][0] - best_loss) / log_data['losses'][0] * 100):.2f}%")

    # Reconstruction accuracy analysis
    recon_data = log_data['reconstruction']
    if recon_data['epoch']:
        report.append(f"\n[Reconstruction Accuracy Analysis]")
        for t_step in ['20', '50', '100', '200']:
            if recon_data[f't{t_step}']:
                avg_acc = np.mean(recon_data[f't{t_step}'])
                final_acc = recon_data[f't{t_step}'][-1]
                report.append(f"t={t_step}: average={avg_acc:.4f}, final={final_acc:.4f}")

    # Generation quality analysis
    div_data = log_data['diversity']
    if div_data['epoch']:
        report.append(f"\n[Generation Quality Analysis]")
        avg_diversity = np.mean(div_data['diversity_score'])
        avg_valid_ratio = np.mean([v/t for v, t in zip(div_data['valid_sequences'], div_data['total_sequences'])])
        avg_length = np.mean(div_data['avg_length'])

        report.append(f"Average diversity score: {avg_diversity:.4f}")
        report.append(f"Average valid sequence rate: {avg_valid_ratio:.4f}")
        report.append(f"Average sequence length: {avg_length:.1f}")

    # Early stopping analysis
    if log_data['patience_counts']:
        max_patience = max([item[1] for item in log_data['patience_counts']])
        report.append(f"\n[Early Stopping Analysis]")
        report.append(f"Maximum patience count: {max_patience}/30")
        report.append(f"Early stopping triggered: {'Yes' if total_epochs < 200 else 'No'}")

    report.append("\n" + "=" * 60)

    # Save report
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"Training summary report saved to: {save_path}")

    # Print report
    for line in report:
        print(line)


def analyze_training_from_log(log_file_path: str, output_dir: str = "./analysis_results/"):
    """Analyze training process from log file and generate all figures"""
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Parse log
    print("Parsing training log...")
    log_data = parse_training_log(log_file_path)

    # Generate various figures
    print("Generating training metrics plot...")
    plot_training_metrics(log_data, f"{output_dir}/training_metrics.png")

    print("Generating reconstruction accuracy heatmap...")
    plot_reconstruction_heatmap(log_data, f"{output_dir}/reconstruction_heatmap.png")

    print("Generating early stopping analysis...")
    plot_early_stopping_analysis(log_data, f"{output_dir}/early_stopping_analysis.png")

    print("Generating training summary report...")
    generate_training_summary_report(log_data, f"{output_dir}/training_summary.txt")

    print(f"All analysis results saved to: {output_dir}")

    return log_data


def analyze_training_from_text(log_content: str, output_dir: str = "./analysis_results/"):
    """Analyze training process from log text and generate all figures"""
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Parse log
    print("Parsing training log...")
    log_data = parse_training_log_from_text(log_content)

    # Generate various figures
    print("Generating training metrics plot...")
    plot_training_metrics(log_data, f"{output_dir}/training_metrics.png")

    print("Generating reconstruction accuracy heatmap...")
    plot_reconstruction_heatmap(log_data, f"{output_dir}/reconstruction_heatmap.png")

    print("Generating early stopping analysis...")
    plot_early_stopping_analysis(log_data, f"{output_dir}/early_stopping_analysis.png")

    print("Generating training summary report...")
    generate_training_summary_report(log_data, f"{output_dir}/training_summary.txt")

    print(f"All analysis results saved to: {output_dir}")

    return log_data


if __name__ == "__main__":
    # Example 1: Analyze AMP length distribution
    try:
        plot_amp_length_distribution("./preprocessed_data/classify.csv")
    except Exception as e:
        print(f"Error plotting AMP length distribution: {e}")

    # Example 2: Analyze training process from log file
    # analyze_training_from_log("training.log", "./analysis_results/")

    # If you have a training log file, use:
    print("\nTo analyze training log, use:")
    print("analyze_training_from_log('your_training_log.txt', './analysis_results/')")
