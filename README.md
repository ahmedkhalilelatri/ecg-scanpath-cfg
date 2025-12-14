# ECG Scanpath Recognition Using Context-Free Grammars

**Computational Theory Course Project - Fall 2025**  
**Mohammed VI Polytechnic University**

## Overview

This project applies Context-Free Grammars (CFG) to recognize expert ECG reading patterns from eye-tracking scanpaths. We built a grammar with 165 production rules and 80 terminals that can distinguish expert from novice ECG interpretation strategies with high accuracy.

## Results

- **95.6% overall accuracy** in classifying expert/intermediate/novice scanpaths
- **Expert recognition**: 100/100 (100%)
- **Intermediate recognition**: 40/50 (80%)
- **Novice rejection**: 99/100 (99%)
- **Parse tree depth**: Expert mean 8.02 ± 1.68 levels, Intermediate/Novice 0.00

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
git clone https://github.com/ahmedkhalilelatri/ecg-scanpath-cfg.git
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
from src.parser_v2 import ECGScanpathParserV2

# Initialize parser with grammar
parser = ECGScanpathParserV2('grammar/grammar_v1.txt')

# Expert scanpath example
expert_scanpath = "II-Rhythm II-Rate I-Axis aVF-Axis V1-Rhythm II-P V1-P II-QRS V1-QRS V2-QRS II-ST V3-ST II-T V3-T II-QT II"

# Parse and classify
result = parser.parse(expert_scanpath)

print(f"Parse successful: {result['success']}")
print(f"Expertise level: {result['expertise']}")
print(f"Parse tree depth: {result['depth']}")
print(f"Partial score: {result['partial_score']:.2f}")
```

**Output:**
```
Parse successful: True
Expertise level: expert
Parse tree depth: 8
Partial score: 0.85
```

### Evaluate on Full Dataset

```bash
python src/evaluate_v2.py
```

This will:
- Load 250 scanpaths (100 expert, 50 intermediate, 100 novice)
- Parse all scanpaths using the CFG with partial scoring
- Generate classification metrics
- Save confusion matrix and depth distribution plots to `results_v2/`

---

## Project Structure

```
ecg-scanpath-cfg/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/
│   ├── parser.py                      # Original CFG parser
│   ├── parser_v2.py                   # Enhanced parser with partial scoring
│   ├── evaluate.py                    # Original evaluation script
│   ├── evaluate_v2.py                 # Enhanced evaluation with partial scoring
│   └── generate_dataset.py            # Dataset generation script
├── grammar/
│   └── grammar_v1.txt                 # Complete 165-rule CFG (80 terminals)
├── data/
│   └── processed/
│       └── all_scanpaths.csv          # 250 synthetic scanpaths
├── results/
│   ├── confusion_matrix.png           # Original results (70% accuracy)
│   ├── depth_distribution.png         # Original depth analysis
│   └── detailed_results.csv           # Original per-scanpath results
├── results_v2/
│   ├── confusion_matrix_v2.png        # Updated results (95.6% accuracy)
│   ├── depth_and_score_distribution_v2.png  # Depth + partial scores
│   └── detailed_results_v2.csv        # Updated per-scanpath results
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
from src.parser_v2 import ECGScanpathParserV2

parser = ECGScanpathParserV2('grammar/grammar_v1.txt')

# Novice example (will fail to parse but get low partial score)
novice = "V3 aVL V6 III V1"
result = parser.parse(novice)
# result['success'] = False, result['expertise'] = 'novice', result['partial_score'] ~ 0.25
```

### 3. Full Evaluation with Metrics

```bash
python src/evaluate_v2.py
```

Outputs:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix (saved as PNG)
- Parse tree depth statistics
- Partial score statistics
- Kruskal-Wallis test results

---

## Key Features

### Partial Parse Scoring
Version 2 introduces partial parse scoring to address the binary limitation of CFG parsing:
- **Full parse** → Expert (score ≥ 0.50)
- **Partial parse** → Intermediate (score 0.10-0.49 with systematic patterns)
- **Failed parse** → Novice (score < 0.10 or random patterns)

This enables graded expertise assessment suitable for medical education contexts.

### Grammar Expansion
The grammar was expanded from 67 to 80 terminals based on comprehensive clinical guidelines:
- Multi-lead P-wave analysis (III-P, V2-V6-P, aVF-aVR-P)
- Extended QT measurement (V3-QT)
- Comprehensive PR interval assessment (I-PR)
- Complete augmented limb lead examination (aVR-ST, aVR-T)

All additions are justified by clinical literature (Dubin 2000, Surawicz 2008).

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

### Terminal Alphabet (80 symbols)

Fixations are represented as `{Lead}-{Component}`:
- **Leads**: I, II, III, aVR, aVL, aVF, V1-V6 (12 leads)
- **Components**: Rhythm, Rate, Axis, P, PR, QRS, ST, T, QT (9 components)

Example: `II-Rhythm`, `V3-ST`, `I-Axis`

Plus 12 plain lead symbols (I, II, ..., V6) for final regional sweep phase.

---

## Grammar Specification

### Context-Free Grammar Definition

**G = (V, Σ, R, S)** where:

- **V** (Non-terminals, 24 total): S, RhythmFirstStrategy, MorphologyFirstStrategy, RegionalFirstStrategy, InitialAssessment, ComponentPhase, RegionalSweep, ...
- **Σ** (Terminals, 80 symbols): Clinically-observed Lead-Component pairs + plain leads
- **R** (Production rules, 165 total): See `grammar/grammar_v1.txt`
- **S** (Start symbol): Root of grammar

### Example Production Rules

```
S -> RhythmFirstStrategy | MorphologyFirstStrategy | RegionalFirstStrategy
RhythmFirstStrategy -> InitialAssessment ExtendedRhythmPhase ComponentPhase RegionalSweep
InitialAssessment -> RhythmCheck RateCheck AxisCheck
RhythmCheck -> 'II-Rhythm'
AxisCheck -> 'I-Axis' 'aVF-Axis'
PWaveLead -> 'II-P' | 'V1-P' | 'I-P' | 'III-P' | 'V2-P' | ... | 'aVR-P'
```

**Complete specification**: See `grammar/grammar_v1.txt`

---

## Results

### Classification Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Expert | 1.000 | 1.000 | 1.000 | 100 |
| Intermediate | 0.976 | 0.800 | 0.879 | 50 |
| Novice | 0.908 | 0.990 | 0.947 | 100 |
| **Overall** | **0.961** | **0.956** | **0.956** | **250** |

**Overall Accuracy: 95.6%** (239/250 correct)

### Per-Class Recognition

- **Expert**: 100/100 (100%)
- **Intermediate**: 40/50 (80%)
- **Novice**: 99/100 (99%)

### Parse Tree Depth and Partial Score Statistics

| Expertise | Depth Mean ± SD | Partial Score Mean ± SD | Parsed |
|-----------|-----------------|-------------------------|--------|
| Expert | 8.02 ± 1.68 | 0.70 ± 0.29 | 100% |
| Intermediate | 0.00 ± 0.00 | 0.13 ± 0.04 | 0% |
| Novice | 0.00 ± 0.00 | 0.29 ± 0.07 | 0% |

**Statistical significance**: Kruskal-Wallis H-test, H = 204.04, p < 10⁻⁴⁴

### Results Visualizations

See `results_v2/` folder for:
- `confusion_matrix_v2.png` - Classification performance
- `depth_and_score_distribution_v2.png` - Parse tree depth + partial scores by expertise level

---

## Theoretical Contribution

### Theorem: Expert ECG Reading is Context-Free but Not Regular

**Theorem**: The language L_expert of expert ECG scanpaths is context-free but not regular.

**Proof Summary**: 
- **Part 1**: L_expert is context-free by construction (we built a CFG generating it)
- **Part 2**: L_expert is not regular, proven via Pumping Lemma for regular languages

Expert ECG reading exhibits hierarchical nesting (Overview → Detail → Verify) that creates unbalanced structures when "pumped", violating the regular language property.

**Implication**: Expert ECG reading exhibits hierarchical nesting that finite automata cannot model. CFG's stack-based derivation is necessary and appropriate.

**See paper** (`paper/ecg_scanpaths_final.pdf`) for complete proof.

---

## Complexity

- **Time**: O(n³|G|) using NLTK Chart Parser (Earley algorithm)
- **Average runtime**: <15ms per scanpath
- **Space**: O(n²|V|)

---

## Paper

Full research paper: [ecg_scanpaths_final.pdf](paper/ecg_scanpaths_final.pdf)

**Abstract**: We present the first application of formal language theory to modeling expert ECG interpretation through eye-tracking scanpaths. Using a CFG with 165 production rules and 80 terminals, we achieve 95.6% accuracy with 100% expert recognition and 80% intermediate recognition. Our work bridges computational theory and cognitive science, proving expert reasoning is context-free but not regular.

---

## Acknowledgments

We used Claude AI (Anthropic) for assistance with code debugging, LaTeX formatting, and literature search. All formal proofs, grammar design, experimental design, and result interpretation are original contributions by the authors.

---

**Course**: Computational Theory (Fall 2025)  
**Institution**: UM6P College of Computing, Rabat, Morocco  
**Instructor**: Professor Mohammed Tahri Sqalli

---

## License

This project is part of academic coursework at Mohammed VI Polytechnic University.

## Citation

If you use this work, please cite:

```bibtex
@article{aitlahcen2025ecg,
  title={Hierarchical Recognition of Expert ECG Interpretation Strategies Using Context-Free Grammars},
  author={Ait Lahcen, Mohamed and Ait Aadi, Youssef and El Atri, Ahmed Khalil},
  institution={Mohammed VI Polytechnic University},
  year={2025}
}
```
