import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import ECGScanpathParser

print("="*70)
print("ECG SCANPATH EVALUATION")
print("="*70)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('data/processed/all_scanpaths.csv')

print(f"Total: {len(df)} scanpaths")
print(f"\nExpertise distribution:")
print(df['expertise_level'].value_counts())

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                      stratify=df['expertise_level'])

print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")

# Initialize parser
print("\nInitializing parser...")
parser = ECGScanpathParser('grammar/grammar_v1.txt')

# Evaluate
print("\n" + "="*70)
print("EVALUATING ON TEST SET")
print("="*70)

predictions = []
true_labels = []
parse_depths = []
strategies = []

for idx, row in test_df.iterrows():
    scanpath = row['fixation_sequence']
    true_label = row['expertise_level']
    
    result = parser.parse(scanpath)
    
    predictions.append(result['expertise'])
    true_labels.append(true_label)
    parse_depths.append(result['depth'])
    strategies.append(result['strategy'])

# Accuracy
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average='macro', zero_division=0
)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Precision (macro): {precision:.3f}")
print(f"Recall (macro): {recall:.3f}")
print(f"F1-Score (macro): {f1:.3f}")

# Per-class metrics
print("\nPer-class performance:")
for label in ['expert', 'intermediate', 'novice']:
    tp = sum(1 for t, p in zip(true_labels, predictions) if t == label and p == label)
    total = sum(1 for t in true_labels if t == label)
    if total > 0:
        print(f"  {label.capitalize()}: {tp}/{total} = {tp/total*100:.1f}%")

# Depth statistics
print("\n" + "="*70)
print("PARSE TREE DEPTH ANALYSIS")
print("="*70)

results_df = pd.DataFrame({
    'true_label': true_labels,
    'predicted': predictions,
    'depth': parse_depths,
    'strategy': strategies
})

for level in ['expert', 'intermediate', 'novice']:
    depths = results_df[results_df['true_label'] == level]['depth']
    if len(depths) > 0:
        print(f"\n{level.capitalize()}:")
        print(f"  Mean depth: {depths.mean():.2f} ± {depths.std():.2f}")
        print(f"  Range: [{depths.min()}, {depths.max()}]")
        print(f"  Parsed: {sum(depths > 0)}/{len(depths)} ({sum(depths > 0)/len(depths)*100:.1f}%)")

# Statistical test
expert_depths = results_df[results_df['true_label'] == 'expert']['depth']
inter_depths = results_df[results_df['true_label'] == 'intermediate']['depth']
novice_depths = results_df[results_df['true_label'] == 'novice']['depth']

expert_depths_nonzero = expert_depths[expert_depths > 0]
inter_depths_nonzero = inter_depths[inter_depths > 0]
novice_depths_nonzero = novice_depths[novice_depths > 0]

if len(expert_depths_nonzero) > 0 and len(novice_depths_nonzero) > 0:
    h_stat, p_value = kruskal(expert_depths_nonzero, inter_depths_nonzero, 
                               novice_depths_nonzero)
    print(f"\nKruskal-Wallis H-test: H={h_stat:.2f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("✓ Significant difference in depth across expertise levels!")

# Create results directory
os.makedirs('results', exist_ok=True)

# Confusion matrix
cm = confusion_matrix(true_labels, predictions, 
                     labels=['expert', 'intermediate', 'novice'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Expert', 'Intermediate', 'Novice'],
            yticklabels=['Expert', 'Intermediate', 'Novice'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300)
print("\n✓ Confusion matrix saved: results/confusion_matrix.png")

# Depth distribution
plt.figure(figsize=(8, 6))
results_df.boxplot(column='depth', by='true_label')
plt.xlabel('Expertise Level')
plt.ylabel('Parse Tree Depth')
plt.title('Parse Tree Depth by Expertise Level')
plt.suptitle('')
plt.tight_layout()
plt.savefig('results/depth_distribution.png', dpi=300)
print("✓ Depth distribution saved: results/depth_distribution.png")

# Save detailed results
results_df.to_csv('results/detailed_results.csv', index=False)
print("✓ Detailed results saved: results/detailed_results.csv")

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)