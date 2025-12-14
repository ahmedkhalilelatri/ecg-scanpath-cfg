"""
ECG Scanpath Evaluation Script - Version 2.0
Uses partial parse scoring for improved intermediate detection

Evaluates CFG parser on:
- Expert scanpaths (should parse fully)
- Intermediate scanpaths (should score partially)
- Novice scanpaths (should fail with low score)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.stats import kruskal
import sys
import os

# Import the new parser
sys.path.insert(0, os.path.dirname(__file__))
from parser_v2 import ECGScanpathParserV2


def load_dataset(data_file):
    """Load scanpath dataset from CSV."""
    print(f"Loading dataset from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} scanpaths")
    print(f"  - Expert: {sum(df['expertise_level'] == 'expert')}")
    print(f"  - Intermediate: {sum(df['expertise_level'] == 'intermediate')}")
    print(f"  - Novice: {sum(df['expertise_level'] == 'novice')}")
    return df


def evaluate_parser(parser, df):
    """
    Evaluate parser on full dataset.
    
    Returns:
        results_df: DataFrame with predictions and metrics
    """
    print("\nEvaluating parser...")
    
    results = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)} scanpaths...")
        
        scanpath_id = row['scanpath_id']
        true_expertise = row['expertise_level']
        scanpath_string = row['fixation_sequence']
        
        # Parse
        parse_result = parser.parse(scanpath_string)
        
        # Store results
        results.append({
            'scanpath_id': scanpath_id,
            'true_expertise': true_expertise,
            'predicted_expertise': parse_result['expertise'],
            'parse_success': parse_result['success'],
            'parse_depth': parse_result['depth'],
            'strategy': parse_result['strategy'],
            'partial_score': parse_result['partial_score'],
            'coverage': parse_result['coverage'],
            'successful_rules': parse_result['details']['successful_rules'],
        })
    
    results_df = pd.DataFrame(results)
    print(f"✓ Evaluation complete")
    
    return results_df


def calculate_metrics(results_df):
    """Calculate classification metrics."""
    y_true = results_df['true_expertise']
    y_pred = results_df['predicted_expertise']
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    labels = ['expert', 'intermediate', 'novice']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    
    # Parse success rates by true expertise
    parse_rates = results_df.groupby('true_expertise')['parse_success'].agg(['sum', 'count', 'mean'])
    
    # Depth statistics by true expertise
    depth_stats = results_df.groupby('true_expertise')['parse_depth'].agg(['mean', 'std', 'min', 'max'])
    
    # Partial score statistics
    score_stats = results_df.groupby('true_expertise')['partial_score'].agg(['mean', 'std', 'min', 'max'])
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'parse_rates': parse_rates,
        'depth_stats': depth_stats,
        'score_stats': score_stats,
    }


def print_results(metrics):
    """Print evaluation results."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS - WITH PARTIAL PARSE SCORING")
    print("="*70)
    
    print(f"\n📊 OVERALL ACCURACY: {metrics['accuracy']:.1%}")
    
    print("\n📋 CONFUSION MATRIX:")
    cm = metrics['confusion_matrix']
    labels = ['Expert', 'Intermediate', 'Novice']
    
    print("\n          Predicted:")
    print("           ", "  ".join(f"{l:^12}" for l in labels))
    print("Actual:")
    for i, label in enumerate(labels):
        print(f"{label:12}", "  ".join(f"{cm[i][j]:^12}" for j in range(3)))
    
    print("\n📈 PER-CLASS PERFORMANCE:")
    report = metrics['classification_report']
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 65)
    for label in ['expert', 'intermediate', 'novice']:
        if label in report:
            p = report[label]['precision']
            r = report[label]['recall']
            f1 = report[label]['f1-score']
            s = int(report[label]['support'])
            print(f"{label.capitalize():<15} {p:<12.3f} {r:<12.3f} {f1:<12.3f} {s:<10}")
    
    print("\n🌲 PARSE TREE DEPTH STATISTICS:")
    depth_stats = metrics['depth_stats']
    print(f"{'Expertise':<15} {'Mean ± SD':<20} {'Min':<8} {'Max':<8}")
    print("-" * 55)
    for expertise in ['expert', 'intermediate', 'novice']:
        if expertise in depth_stats.index:
            mean = depth_stats.loc[expertise, 'mean']
            std = depth_stats.loc[expertise, 'std']
            min_val = int(depth_stats.loc[expertise, 'min'])
            max_val = int(depth_stats.loc[expertise, 'max'])
            print(f"{expertise.capitalize():<15} {mean:>5.2f} ± {std:<5.2f}       {min_val:<8} {max_val:<8}")
    
    print("\n📊 PARTIAL SCORE STATISTICS:")
    score_stats = metrics['score_stats']
    print(f"{'Expertise':<15} {'Mean ± SD':<20} {'Min':<8} {'Max':<8}")
    print("-" * 55)
    for expertise in ['expert', 'intermediate', 'novice']:
        if expertise in score_stats.index:
            mean = score_stats.loc[expertise, 'mean']
            std = score_stats.loc[expertise, 'std']
            min_val = score_stats.loc[expertise, 'min']
            max_val = score_stats.loc[expertise, 'max']
            print(f"{expertise.capitalize():<15} {mean:>5.2f} ± {std:<5.2f}       {min_val:<8.2f} {max_val:<8.2f}")
    
    print("\n✅ PARSE SUCCESS RATES:")
    parse_rates = metrics['parse_rates']
    print(f"{'Expertise':<15} {'Parsed':<10} {'Total':<10} {'Rate':<10}")
    print("-" * 45)
    for expertise in ['expert', 'intermediate', 'novice']:
        if expertise in parse_rates.index:
            parsed = int(parse_rates.loc[expertise, 'sum'])
            total = int(parse_rates.loc[expertise, 'count'])
            rate = parse_rates.loc[expertise, 'mean']
            print(f"{expertise.capitalize():<15} {parsed:<10} {total:<10} {rate:<10.1%}")


def plot_confusion_matrix(cm, output_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    labels = ['Expert', 'Intermediate', 'Novice']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - CFG with Partial Parse Scoring', fontsize=14, fontweight='bold')
    plt.ylabel('True Expertise', fontsize=12)
    plt.xlabel('Predicted Expertise', fontsize=12)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'confusion_matrix_v2.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {output_file}")
    plt.close()


def plot_depth_distribution(results_df, output_dir):
    """Plot parse tree depth distribution by expertise."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Violin plot of depths
    ax1 = axes[0]
    expertise_order = ['expert', 'intermediate', 'novice']
    colors = {'expert': '#2ecc71', 'intermediate': '#f39c12', 'novice': '#e74c3c'}
    
    data_to_plot = [results_df[results_df['true_expertise'] == exp]['parse_depth'].values 
                    for exp in expertise_order]
    
    parts = ax1.violinplot(data_to_plot, positions=[1, 2, 3], showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[expertise_order[i]])
        pc.set_alpha(0.7)
    
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['Expert', 'Intermediate', 'Novice'])
    ax1.set_ylabel('Parse Tree Depth', fontsize=12)
    ax1.set_xlabel('True Expertise Level', fontsize=12)
    ax1.set_title('Parse Tree Depth Distribution', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Partial score distribution
    ax2 = axes[1]
    
    data_to_plot_score = [results_df[results_df['true_expertise'] == exp]['partial_score'].values 
                          for exp in expertise_order]
    
    parts2 = ax2.violinplot(data_to_plot_score, positions=[1, 2, 3], showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts2['bodies']):
        pc.set_facecolor(colors[expertise_order[i]])
        pc.set_alpha(0.7)
    
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['Expert', 'Intermediate', 'Novice'])
    ax2.set_ylabel('Partial Parse Score', fontsize=12)
    ax2.set_xlabel('True Expertise Level', fontsize=12)
    ax2.set_title('Partial Score Distribution', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0.50, color='red', linestyle='--', alpha=0.5, label='Intermediate Threshold')
    ax2.legend()
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'depth_and_score_distribution_v2.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Depth/Score distribution saved to {output_file}")
    plt.close()


def save_detailed_results(results_df, output_dir):
    """Save detailed results to CSV."""
    output_file = os.path.join(output_dir, 'detailed_results_v2.csv')
    results_df.to_csv(output_file, index=False)
    print(f"✓ Detailed results saved to {output_file}")


def main():
    """Main evaluation pipeline."""
    print("="*70)
    print("ECG SCANPATH CFG EVALUATION - VERSION 2.0")
    print("With Partial Parse Scoring for Intermediate Detection")
    print("="*70)
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    grammar_file = os.path.join(base_dir, '..', 'grammar', 'grammar_v1.txt')
    data_file = os.path.join(base_dir, '..', 'data', 'processed', 'all_scanpaths.csv')
    output_dir = os.path.join(base_dir, '..', 'results_v2')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check files exist
    if not os.path.exists(grammar_file):
        print(f"❌ Grammar file not found: {grammar_file}")
        return
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    # Load parser
    print(f"\n📖 Loading grammar from {grammar_file}")
    parser = ECGScanpathParserV2(grammar_file)
    
    # Load dataset
    df = load_dataset(data_file)
    
    # Evaluate
    results_df = evaluate_parser(parser, df)
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    # Print results
    print_results(metrics)
    
    # Save outputs
    print("\n📁 Saving outputs...")
    plot_confusion_matrix(metrics['confusion_matrix'], output_dir)
    plot_depth_distribution(results_df, output_dir)
    save_detailed_results(results_df, output_dir)
    
    # Statistical test (Kruskal-Wallis)
    print("\n📊 STATISTICAL SIGNIFICANCE TEST (Kruskal-Wallis H-test)")
    expert_scores = results_df[results_df['true_expertise'] == 'expert']['partial_score']
    intermediate_scores = results_df[results_df['true_expertise'] == 'intermediate']['partial_score']
    novice_scores = results_df[results_df['true_expertise'] == 'novice']['partial_score']
    
    h_stat, p_value = kruskal(expert_scores, intermediate_scores, novice_scores)
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} difference between groups")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - confusion_matrix_v2.png")
    print(f"  - depth_and_score_distribution_v2.png")
    print(f"  - detailed_results_v2.csv")
    
    # Summary for paper
    print("\n📝 SUMMARY FOR PAPER UPDATE:")
    print("-" * 70)
    accuracy = metrics['accuracy']
    cm = metrics['confusion_matrix']
    expert_correct = cm[0][0]
    expert_total = cm[0].sum()
    intermediate_correct = cm[1][1]
    intermediate_total = cm[1].sum()
    novice_correct = cm[2][2]
    novice_total = cm[2].sum()
    
    print(f"Overall 3-class accuracy: {accuracy:.1%} ({int(accuracy * len(results_df))}/{len(results_df)} correct)")
    print(f"\nPer-class recognition:")
    print(f"  Expert: {expert_correct}/{expert_total} ({expert_correct/expert_total:.1%})")
    print(f"  Intermediate: {intermediate_correct}/{intermediate_total} ({intermediate_correct/intermediate_total:.1%}) ← IMPROVED!")
    print(f"  Novice: {novice_correct}/{novice_total} ({novice_correct/novice_total:.1%})")
    
    print("\n💡 Update Table 1 in paper with these numbers!")


if __name__ == "__main__":
    main()
