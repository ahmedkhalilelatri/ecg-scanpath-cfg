# ECG Scanpath Recognition Using Context-Free Grammars

**Computational Theory Course Project - Fall 2025**  
**Mohammed VI Polytechnic University**

## Overview

This project applies Context-Free Grammars (CFG) to recognize expert ECG reading patterns from eye-tracking scanpaths. We built a grammar with 132 production rules that can distinguish expert from novice ECG interpretation strategies.

## Results

- **70% overall accuracy** in classifying expert/intermediate/novice scanpaths
- **87.5% accuracy** for binary classification (expert vs. non-expert)
- **Expert recognition**: 15/20 (75%)
- **Novice rejection**: 20/20 (100%)
- **Parse tree depth**: Expert mean 5.55 ± 3.39 levels, Novice/Intermediate 0.00

## Authors

- **Mohamed Ait Lahcen** - Mohammed VI Polytechnic University
- **Youssef Ait Aadi** - Mohammed VI Polytechnic University
- **Ahmed Khalil El Atri** - Mohammed VI Polytechnic University

**Course**: Computational Theory (Fall 2025)  
**Institution**: UM6P College of Computing, Rabat, Morocco

---

## Installation

**Requirements**: Python 3.10 or higher

```bash
# Clone the repository
git clone https://github.com/yourusername/ecg-scanpath-cfg.git
cd ecg-scanpath-cfg

# Install dependencies
pip install -r requirements.txt

# Test it works
python examples/demo.py
```

---

## Quick Start

### Parse a Single Scanpath

```python
from src.parser import ECGScanpathParser

# Initialize parser with grammar
parser = ECGScanpathParser('grammar/grammar_v1.txt')

# Expert scanpath example
expert_scanpath = "II-Rhythm II-Rate I-Axis aVF-Axis V1-Rhythm II-P V1-P II-QRS V1-QRS V2-QRS II-ST V3-ST II-T V3-T II-QT II"

# Parse and classify
result = parser.parse(expert_scanpath)

print(f"Parse successful: {result['success']}")
print(f"Expertise level: {result['expertise']}")
print(f"Parse tree depth: {result['depth']}")
print(f"Strategy detected: {result['strategy']}")
```

**Output:**
```
Parse successful: True
Expertise level: expert
Parse tree depth: 6
Strategy detected: RhythmFirstStrategy
```

### Evaluate on Full Dataset

```bash
python src/evaluate.py
```

This will:
- Load 250 scanpaths (100 expert, 50 intermediate, 100 novice)
- Parse all scanpaths using the CFG
- Generate classification metrics
- Save confusion matrix and depth distribution plots to `results/`

---

## Project Structure

```
ecg-scanpath-cfg/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/
│   ├── parser.py                      # CFG parser implementation
│   ├── evaluate.py                    # Evaluation script
│   └── generate_dataset.py            # Dataset generation script
├── grammar/
│   └── grammar_v1.txt                 # Complete 132-rule CFG specification
├── data/
│   └── processed/
│       └── all_scanpaths.csv          # 250 synthetic scanpaths
├── results/
│   ├── confusion_matrix.png           # Classification results visualization
│   ├── depth_distribution.png         # Parse tree depth by expertise
│   └── detailed_results.csv           # Per-scanpath results
├── paper/
│   └── ecg_scanpaths_final.pdf        # Full research paper (ACM format)
└── examples/
    └── demo.py                        # Simple usage demonstration
```

---

## Usage

### 1. Generate Synthetic Dataset

```bash
python src/generate_dataset.py
```

Creates `data/processed/all_scanpaths.csv` with:
- 100 expert scanpaths (3 strategies: Rhythm-First, Morphology-First, Regional-First)
- 50 intermediate scanpaths (partial systematic patterns)
- 100 novice scanpaths (random, unsystematic)

### 2. Parse Individual Scanpaths

```python
from src.parser import ECGScanpathParser

parser = ECGScanpathParser('grammar/grammar_v1.txt')

# Novice example (will fail to parse)
novice = "V3 aVL V6 III V1"
result = parser.parse(novice)
# result['success'] = False, result['expertise'] = 'novice'
```

### 3. Full Evaluation with Metrics

```bash
python src/evaluate.py
```

Outputs:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix (saved as PNG)
- Parse tree depth statistics
- Kruskal-Wallis test results

---

## Dataset

### Format

The dataset (`data/processed/all_scanpaths.csv`) contains:

```csv
scanpath_id,expertise_level,strategy_type,ecg_scenario,fixation_sequence,sequence_length,has_abnormality
SCAN_EXP_001,expert,rhythm_first,normal_sinus,II-Rhythm II-Rate I-Axis aVF-Axis ...,33,False
SCAN_NOV_001,novice,random,normal_sinus,V3 II aVL V6 III V1 aVR V4 ...,20,False
```

### ECG Scenarios

- **Normal Sinus Rhythm** (40%)
- **Anterior STEMI** (20%)
- **Atrial Fibrillation** (15%)
- **Inferior MI** (15%)
- **Bundle Branch Block** (10%)

### Terminal Alphabet (108 symbols)

Fixations are represented as `{Lead}-{Component}`:
- **Leads**: I, II, III, aVR, aVL, aVF, V1-V6 (12 leads)
- **Components**: Rhythm, Rate, Axis, P, PR, QRS, ST, T, QT (9 components)

Example: `II-Rhythm`, `V3-ST`, `I-Axis`

---

## Grammar Specification

### Context-Free Grammar Definition

**G = (V, Σ, R, S)** where:

- **V** (Non-terminals, 24 total): S, RhythmFirstStrategy, MorphologyFirstStrategy, RegionalFirstStrategy, InitialAssessment, ComponentPhase, RegionalSweep, ...
- **Σ** (Terminals, 108 symbols): All Lead-Component pairs
- **R** (Production rules, 132 total): See `grammar/grammar_v1.txt`
- **S** (Start symbol): Root of grammar

### Example Production Rules

```
S -> RhythmFirstStrategy | MorphologyFirstStrategy | RegionalFirstStrategy
RhythmFirstStrategy -> InitialAssessment ExtendedRhythmPhase ComponentPhase RegionalSweep
InitialAssessment -> RhythmCheck RateCheck AxisCheck
RhythmCheck -> 'II-Rhythm'
AxisCheck -> 'I-Axis' 'aVF-Axis'
```

**Complete specification**: See `grammar/grammar_v1.txt`

---

## Results

### Classification Performance

| Metric | 3-Class | Binary (Expert vs. Non-Expert) |
|--------|---------|--------------------------------|
| **Accuracy** | 70.0% | **87.5%** |
| **Precision** | 0.524 | 0.875 |
| **Recall** | 0.583 | 0.875 |
| **F1-Score** | 0.528 | 0.875 |

### Per-Class Recognition

- **Expert**: 15/20 (75%)
- **Intermediate**: 0/10 (0%)
- **Novice**: 20/20 (100%)

### Parse Tree Depth Statistics

| Expertise | Mean Depth ± SD | Min | Max | Parsed |
|-----------|-----------------|-----|-----|--------|
| Expert | 5.55 ± 3.39 | 0 | 10 | 75% |
| Intermediate | 0.00 ± 0.00 | 0 | 0 | 0% |
| Novice | 0.00 ± 0.00 | 0 | 0 | 0% |

**Statistical significance**: Kruskal-Wallis H-test, p < 0.001

### Results Visualizations

See `results/` folder for:
- `confusion_matrix.png` - Classification performance
- `depth_distribution.png` - Parse tree depth by expertise level

---

## Theoretical Contribution

### Theorem: Expert ECG Reading is Context-Free but Not Regular

**Theorem**: The language L_expert of expert ECG scanpaths is context-free but not regular.

**Proof Summary**: 
- **Part 1**: L_expert is context-free by construction (we built a CFG generating it)
- **Part 2**: L_expert is not regular, proven via Pumping Lemma for regular languages

**Implication**: Expert ECG reading exhibits hierarchical nesting that finite automata cannot model. CFG's stack-based derivation is necessary and appropriate.

**See paper** (`paper/ecg_scanpaths_final.pdf`) for complete proof.

---

### Complexity

- **Time**: O(n³|G|) using NLTK Chart Parser
- **Average runtime**: <15ms per scanpath
- **Space**: O(n²|V|)

---

## Paper

Full research paper: [ecg_scanpaths_final.pdf](paper/ecg_scanpaths_final.pdf)

---

## Acknowledgments

We used Claude AI (Anthropic) for code debugging, LaTeX formatting, and literature search. All core contributions (grammar design, theoretical proofs, experimental methodology) are original student work.

---

**Course**: Computational Theory (Fall 2025)  
**Institution**: UM6P College of Computing, Rabat, Morocco  
**Instructor**: Professor Mohammed Tahri Sqalli

