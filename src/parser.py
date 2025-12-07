import nltk
from nltk import CFG
from nltk.parse.chart import ChartParser

class ECGScanpathParser:
    def __init__(self, grammar_file='grammar/grammar_v1.txt'):
        with open(grammar_file, 'r', encoding='utf-8') as f:
            grammar_str = f.read()
        
        self.grammar = CFG.fromstring(grammar_str)
        self.parser = ChartParser(self.grammar)
        
        print(f"✓ Grammar loaded: {len(self.grammar.productions())} productions")
    
    def parse(self, scanpath_string):
        tokens = scanpath_string.split()
        
        try:
            trees = list(self.parser.parse(tokens))
            
            if len(trees) == 0:
                return {
                    'success': False,
                    'tree': None,
                    'depth': 0,
                    'strategy': 'none',
                    'expertise': 'novice'
                }
            
            tree = trees[0]
            depth = self._compute_depth(tree)
            strategy = self._identify_strategy(tree)
            expertise = self._classify_expertise(depth)
            
            return {
                'success': True,
                'tree': tree,
                'depth': depth,
                'strategy': strategy,
                'expertise': expertise,
                'num_parses': len(trees)
            }
        
        except Exception as e:
            return {
                'success': False,
                'tree': None,
                'depth': 0,
                'strategy': 'none',
                'expertise': 'novice'
            }
    
    def _compute_depth(self, tree):
        if isinstance(tree, str):
            return 0
        if len(tree) == 0:
            return 0
        return 1 + max(self._compute_depth(child) for child in tree)
    
    def _identify_strategy(self, tree):
        if tree is None:
            return 'none'
        
        for subtree in tree.subtrees():
            label = str(subtree.label())
            if 'RhythmFirst' in label:
                return 'rhythm_first'
            elif 'MorphologyFirst' in label:
                return 'morphology_first'
            elif 'RegionalFirst' in label:
                return 'regional_first'
        
        return 'unknown'
    
    def _classify_expertise(self, depth):
        if depth >= 5:
            return 'expert'
        elif depth >= 3:
            return 'intermediate'
        else:
            return 'novice'

if __name__ == '__main__':
    parser = ECGScanpathParser('grammar/grammar_v1.txt')
    
    print("\nTest 1: Expert pattern")
    expert = "II-Rhythm II-Rate I-Axis aVF-Axis V1-Rhythm II-P V1-P II-QRS V3-QRS II-ST V3-ST II-T V3-T II"
    result = parser.parse(expert)
    print(f"Success: {result['success']}, Depth: {result['depth']}, Strategy: {result['strategy']}")
    
    print("\nTest 2: Novice pattern")
    novice = "V3 aVL V6 III V1"
    result2 = parser.parse(novice)
    print(f"Success: {result2['success']}, Depth: {result2['depth']}")