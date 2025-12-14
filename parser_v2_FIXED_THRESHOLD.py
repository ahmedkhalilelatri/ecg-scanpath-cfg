"""
ECG Scanpath Parser with Partial Parse Scoring
Version 2.0 - Adds intermediate expertise detection

Key Changes:
- Partial parse scoring for failed parses
- Three-level classification: expert / intermediate / novice
- Measures grammar coverage and rule application
"""

import nltk
from nltk import CFG, ChartParser
from typing import Dict, List, Tuple, Optional
import re


class ECGScanpathParserV2:
    """
    Context-Free Grammar parser for ECG scanpaths with partial scoring.
    
    Supports:
    - Full parse detection (expert)
    - Partial parse scoring (intermediate)
    - Parse failure (novice)
    """
    
    def __init__(self, grammar_file: str):
        """
        Initialize parser with grammar file.
        
        Args:
            grammar_file: Path to CFG grammar file (grammar_v1.txt)
        """
        self.grammar = self._load_grammar(grammar_file)
        self.parser = ChartParser(self.grammar)
        self.total_rules = len(self.grammar.productions())
        
        print(f"✓ Grammar loaded: {self.total_rules} production rules")
        print(f"✓ Non-terminals: {len(self.grammar._lhs_index)} types")
        print(f"✓ Terminals: {len(self.grammar._rhs_index)} symbols")
    
    def _load_grammar(self, grammar_file: str) -> CFG:
        """Load CFG from file."""
        with open(grammar_file, 'r', encoding='utf-8') as f:
            grammar_str = f.read()
        return CFG.fromstring(grammar_str)
    
    def _calculate_depth(self, tree) -> int:
        """
        Calculate maximum depth of parse tree.
        
        Args:
            tree: NLTK Tree object
            
        Returns:
            Maximum depth from root to deepest leaf
        """
        if isinstance(tree, str):  # Leaf node (terminal)
            return 0
        
        if len(tree) == 0:  # Empty tree
            return 0
        
        # Recursive case: 1 + max depth of children
        child_depths = [self._calculate_depth(child) for child in tree]
        return 1 + max(child_depths) if child_depths else 0
    
    def _extract_partial_features(self, tokens: List[str]) -> Dict:
        """
        Extract features from partial parse attempts.
        
        This is the KEY innovation - analyzing what happens when parse fails.
        
        Args:
            tokens: List of scanpath tokens
            
        Returns:
            Dictionary of partial parse features
        """
        # Filter tokens to only those covered by grammar (handle missing terminals gracefully)
        valid_tokens = self._filter_valid_tokens(tokens)
        
        if len(valid_tokens) == 0:
            # No valid tokens - return zero scores
            return {
                'successful_rules': 0,
                'partial_score': 0.0,
                'coverage': 0.0,
                'max_span': 0,
                'total_edges': 0,
                'complete_edges': 0,
                'has_initial_rhythm': False,
                'has_axis_check': False,
                'has_component_analysis': False,
            }
        
        try:
            # Create chart for this input
            chart = self.parser.chart_parse(valid_tokens)
        except ValueError as e:
            # Grammar doesn't cover some tokens - return minimal features
            print(f"  Warning: {e}")
            return {
                'successful_rules': 0,
                'partial_score': 0.0,
                'coverage': 0.0,
                'max_span': 0,
                'total_edges': 0,
                'complete_edges': 0,
                'has_initial_rhythm': self._has_subsequence(tokens, ['II-Rhythm', 'II-Rate']),
                'has_axis_check': self._has_subsequence(tokens, ['I-Axis', 'aVF-Axis']),
                'has_component_analysis': any(
                    self._has_subsequence(tokens, [comp])
                    for comp in ['P', 'QRS', 'ST', 'T', 'QT']
                ),
            }
        
        # Count successful edges (partial derivations)
        successful_rules = 0
        max_span_length = 0
        total_edges = 0
        complete_edges = 0
        
        for edge in chart.edges():
            total_edges += 1
            if edge.is_complete():  # Rule successfully applied
                complete_edges += 1
                successful_rules += 1
                
                # Track maximum contiguous span covered
                span_length = edge.end() - edge.start()
                max_span_length = max(max_span_length, span_length)
        
        # Calculate coverage metrics
        input_length = len(tokens)
        coverage = max_span_length / input_length if input_length > 0 else 0
        partial_score = successful_rules / self.total_rules if self.total_rules > 0 else 0
        
        # Check for specific systematic patterns (even in failed parses)
        has_initial_rhythm = self._has_subsequence(tokens, ['II-Rhythm', 'II-Rate'])
        has_axis_check = self._has_subsequence(tokens, ['I-Axis', 'aVF-Axis'])
        has_component_analysis = any(
            self._has_subsequence(tokens, [comp])
            for comp in ['P', 'QRS', 'ST', 'T', 'QT']
        )
        
        return {
            'successful_rules': successful_rules,
            'partial_score': partial_score,
            'coverage': coverage,
            'max_span': max_span_length,
            'total_edges': total_edges,
            'complete_edges': complete_edges,
            'has_initial_rhythm': has_initial_rhythm,
            'has_axis_check': has_axis_check,
            'has_component_analysis': has_component_analysis,
        }
    
    def _filter_valid_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens to only those covered by the grammar.
        
        Args:
            tokens: Input tokens
            
        Returns:
            List of tokens that exist in grammar
        """
        # Get all terminals from grammar
        grammar_terminals = set()
        for production in self.grammar.productions():
            for symbol in production.rhs():
                if isinstance(symbol, str):  # Terminal
                    grammar_terminals.add(symbol)
        
        # Filter to valid tokens
        valid_tokens = [t for t in tokens if t in grammar_terminals]
        
        if len(valid_tokens) < len(tokens):
            missing = set(tokens) - grammar_terminals
            if len(missing) <= 5:  # Only print if not too many
                print(f"  Info: Filtered {len(tokens) - len(valid_tokens)} tokens not in grammar: {list(missing)[:5]}")
        
        return valid_tokens
    
    def _has_subsequence(self, tokens: List[str], pattern: List[str]) -> bool:
        """Check if tokens contain pattern as subsequence (not necessarily contiguous)."""
        pattern_idx = 0
        for token in tokens:
            # Check if this token matches any part of the pattern
            for p in pattern:
                if p in token:  # Fuzzy match (e.g., 'Rhythm' matches 'II-Rhythm')
                    pattern_idx += 1
                    if pattern_idx == len(pattern):
                        return True
                    break
        return False
    
    def _classify_by_partial_score(self, partial_features: Dict, full_parse: bool) -> str:
        """
        Classify expertise level based on partial parse features.
        
        Decision logic:
        1. Full parse → expert
        2. High partial score (≥0.50) OR high coverage (≥0.70) → intermediate
        3. Otherwise → novice
        
        Args:
            partial_features: Features from partial parse analysis
            full_parse: Whether full parse succeeded
            
        Returns:
            'expert', 'intermediate', or 'novice'
        """
        if full_parse:
            return 'expert'
        
        score = partial_features['partial_score']
        coverage = partial_features['coverage']
        has_initial = partial_features['has_initial_rhythm']
        has_axis = partial_features['has_axis_check']
        has_components = partial_features['has_component_analysis']
        
        # Intermediate criteria - adjusted based on empirical data
        # Intermediate scores: mean 0.13, range 0.07-0.21
        # Novice scores: mean 0.29, range 0.16-0.46
        # Expert scores: mean 0.70, range 0.29-1.42
        if score >= 0.50:  # High systematic
            return 'intermediate'
        elif score >= 0.20 and coverage >= 0.30:  # Partial systematic with decent coverage
            return 'intermediate'
        elif has_initial and (has_axis or has_components) and score >= 0.10:  # Started systematically
            return 'intermediate'
        
        # Otherwise novice (random clicking, no systematic pattern)
        return 'novice'
    
    def _detect_strategy(self, tree) -> str:
        """
        Detect which strategy was used from parse tree root.
        
        Args:
            tree: Parse tree from successful parse
            
        Returns:
            Strategy name (RhythmFirstStrategy, MorphologyFirstStrategy, RegionalFirstStrategy)
        """
        if tree is None:
            return 'Unknown'
        
        # Root should be S, first child is strategy
        if len(tree) > 0:
            strategy_node = tree[0]
            if hasattr(strategy_node, 'label'):
                return strategy_node.label()
        
        return 'Unknown'
    
    def parse(self, scanpath_string: str) -> Dict:
        """
        Parse ECG scanpath and classify expertise level.
        
        NEW: Includes partial parse scoring for intermediate detection!
        
        Args:
            scanpath_string: Space-separated scanpath (e.g., "II-Rhythm II-Rate I-Axis ...")
            
        Returns:
            Dictionary with:
            - success: bool (full parse succeeded)
            - expertise: str ('expert', 'intermediate', 'novice')
            - depth: int (parse tree depth, 0 if failed)
            - strategy: str (detected strategy type)
            - partial_score: float (0-1, percentage of grammar utilized)
            - coverage: float (0-1, percentage of input covered)
            - details: dict (additional metrics)
        """
        # Tokenize input
        tokens = scanpath_string.strip().split()
        
        if len(tokens) == 0:
            return {
                'success': False,
                'expertise': 'novice',
                'depth': 0,
                'strategy': 'None',
                'partial_score': 0.0,
                'coverage': 0.0,
                'details': {}
            }
        
        # Attempt full parse
        try:
            # Filter to valid tokens first
            valid_tokens = self._filter_valid_tokens(tokens)
            
            if len(valid_tokens) == 0:
                # No valid tokens at all
                parses = []
                full_parse_success = False
            else:
                parses = list(self.parser.parse(valid_tokens))
                full_parse_success = len(parses) > 0
        except ValueError as e:
            # Grammar coverage error - treat as parse failure
            print(f"  Parse error (coverage): {e}")
            full_parse_success = False
            parses = []
        except Exception as e:
            print(f"  Parse error: {e}")
            full_parse_success = False
            parses = []
        
        # Extract partial features (even for successful parses, for debugging)
        partial_features = self._extract_partial_features(tokens)
        
        # Classify expertise
        expertise = self._classify_by_partial_score(partial_features, full_parse_success)
        
        # Calculate depth and strategy (only for successful parses)
        if full_parse_success and len(parses) > 0:
            tree = parses[0]  # Take first parse if multiple
            depth = self._calculate_depth(tree)
            strategy = self._detect_strategy(tree)
        else:
            depth = 0
            strategy = 'None'
            tree = None
        
        return {
            'success': full_parse_success,
            'expertise': expertise,
            'depth': depth,
            'strategy': strategy,
            'partial_score': partial_features['partial_score'],
            'coverage': partial_features['coverage'],
            'details': {
                'successful_rules': partial_features['successful_rules'],
                'max_span': partial_features['max_span'],
                'total_edges': partial_features['total_edges'],
                'complete_edges': partial_features['complete_edges'],
                'has_initial_rhythm': partial_features['has_initial_rhythm'],
                'has_axis_check': partial_features['has_axis_check'],
                'has_component_analysis': partial_features['has_component_analysis'],
                'input_length': len(tokens),
            }
        }
    
    def parse_batch(self, scanpaths: List[Tuple[str, str]]) -> List[Dict]:
        """
        Parse multiple scanpaths.
        
        Args:
            scanpaths: List of (scanpath_id, scanpath_string) tuples
            
        Returns:
            List of parse results
        """
        results = []
        for idx, (scan_id, scanpath) in enumerate(scanpaths):
            if (idx + 1) % 10 == 0:
                print(f"  Parsed {idx + 1}/{len(scanpaths)} scanpaths...")
            
            result = self.parse(scanpath)
            result['scanpath_id'] = scan_id
            results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    import os
    
    # Test with examples
    grammar_path = os.path.join(os.path.dirname(__file__), 'grammar', 'grammar_v1.txt')
    
    if not os.path.exists(grammar_path):
        print(f"Grammar file not found: {grammar_path}")
        print("Please ensure grammar/grammar_v1.txt exists")
    else:
        parser = ECGScanpathParserV2(grammar_path)
        
        print("\n" + "="*70)
        print("EXAMPLE 1: Expert Scanpath (should parse successfully)")
        print("="*70)
        
        expert = "II-Rhythm II-Rate I-Axis aVF-Axis V1-Rhythm II-P V1-P II-QRS V1-QRS V2-QRS II-ST V3-ST II-T V3-T II-QT II III aVF"
        result = parser.parse(expert)
        
        print(f"Input: {expert[:80]}...")
        print(f"\nResult:")
        print(f"  ✓ Parse success: {result['success']}")
        print(f"  ✓ Expertise: {result['expertise'].upper()}")
        print(f"  ✓ Parse tree depth: {result['depth']}")
        print(f"  ✓ Strategy: {result['strategy']}")
        print(f"  ✓ Partial score: {result['partial_score']:.2f}")
        print(f"  ✓ Coverage: {result['coverage']:.2f}")
        
        print("\n" + "="*70)
        print("EXAMPLE 2: Intermediate Scanpath (partial systematic)")
        print("="*70)
        
        intermediate = "II-Rhythm II-Rate I-Axis aVF-Axis II-P V1-P II-QRS V1-QRS V3 aVF"
        result = parser.parse(intermediate)
        
        print(f"Input: {intermediate}")
        print(f"\nResult:")
        print(f"  ✓ Parse success: {result['success']}")
        print(f"  ✓ Expertise: {result['expertise'].upper()}")
        print(f"  ✓ Partial score: {result['partial_score']:.2f}")
        print(f"  ✓ Coverage: {result['coverage']:.2f}")
        print(f"  ✓ Rules applied: {result['details']['successful_rules']}/{parser.total_rules}")
        
        print("\n" + "="*70)
        print("EXAMPLE 3: Novice Scanpath (random)")
        print("="*70)
        
        novice = "V3 II aVL V6 III V1 aVR V4"
        result = parser.parse(novice)
        
        print(f"Input: {novice}")
        print(f"\nResult:")
        print(f"  ✓ Parse success: {result['success']}")
        print(f"  ✓ Expertise: {result['expertise'].upper()}")
        print(f"  ✓ Partial score: {result['partial_score']:.2f}")
        print(f"  ✓ Coverage: {result['coverage']:.2f}")
        print(f"  ✓ Rules applied: {result['details']['successful_rules']}/{parser.total_rules}")
