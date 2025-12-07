"""
Demo script showing how to use the ECG Scanpath CFG Parser

This script demonstrates:
1. Loading the grammar
2. Parsing expert vs novice scanpaths
3. Interpreting results
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from parser import ECGScanpathParser

def main():
    print("=" * 70)
    print("ECG Scanpath CFG Parser - Demo")
    print("=" * 70)
    print()
    
    # Initialize parser
    print("Loading grammar from grammar/grammar_v1.txt...")
    grammar_path = os.path.join(os.path.dirname(__file__), '..', 'grammar', 'grammar_v1.txt')
    parser = ECGScanpathParser(grammar_path)
    print(f"✓ Grammar loaded successfully ({len(parser.grammar.productions())} rules)")
    print()
    
    # Example 1: Expert scanpath (should parse successfully)
    print("-" * 70)
    print("Example 1: Expert Scanpath (Rhythm-First Strategy)")
    print("-" * 70)
    
    expert_scanpath = (
        "II-Rhythm II-Rate I-Axis aVF-Axis "
        "V1-Rhythm "
        "II-P V1-P "
        "II-QRS V1-QRS V2-QRS "
        "II-ST V3-ST "
        "II-T V3-T "
        "II-QT "
        "II III aVF V1 V2 V3"
    )
    
    print(f"Input: {expert_scanpath}")
    print()
    
    result = parser.parse(expert_scanpath)
    
    print(f"Parse successful: {result['success']}")
    print(f"Classified as: {result['expertise'].upper()}")
    print(f"Parse tree depth: {result['depth']} levels")
    print(f"Strategy detected: {result['strategy']}")
    print()
    
    # Example 2: Novice scanpath (should fail to parse)
    print("-" * 70)
    print("Example 2: Novice Scanpath (Random, Unsystematic)")
    print("-" * 70)
    
    novice_scanpath = "V3 II aVL V6 III V1 aVR V4 V5"
    
    print(f"Input: {novice_scanpath}")
    print()
    
    result = parser.parse(novice_scanpath)
    
    print(f"Parse successful: {result['success']}")
    print(f"Classified as: {result['expertise'].upper()}")
    print(f"Parse tree depth: {result['depth']} levels")
    print(f"Reason: {result.get('reason', 'Does not follow systematic ECG reading pattern')}")
    print()
    
    # Example 3: Show depth as expertise indicator
    print("-" * 70)
    print("Example 3: Parse Tree Depth as Expertise Indicator")
    print("-" * 70)
    print()
    print("Expert scanpaths produce deep parse trees (mean depth: 5.55 ± 3.39)")
    print("because they follow hierarchical cognitive strategies:")
    print()
    print("  Level 1: Strategy Selection (Rhythm-First, Morphology-First, Regional-First)")
    print("  Level 2: Major Phases (Initial Assessment, Component Analysis, Regional Sweep)")
    print("  Level 3: Component Groups (Rhythm Check, P-wave Check, QRS Check, etc.)")
    print("  Level 4: Lead Sequences (Systematic lead-by-lead examination)")
    print("  Level 5: Specific Fixations (e.g., II-Rhythm, V1-P, V3-ST)")
    print()
    print("Novice scanpaths fail to parse (depth = 0) because they lack:")
    print("  - Initial systematic assessment (Rhythm, Rate, Axis)")
    print("  - Component-based analysis (P, QRS, ST, T waves)")
    print("  - Systematic lead ordering")
    print()
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Run full evaluation: python src/evaluate.py")
    print("  2. Generate new datasets: python src/generate_dataset.py")
    print("  3. Read the paper: paper/ecg_scanpaths_final.pdf")
    print()

if __name__ == "__main__":
    main()
