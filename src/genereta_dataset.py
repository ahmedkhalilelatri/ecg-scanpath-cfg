import random
import csv
import os

ALL_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def generate_expert_rhythm_first():
    """Generate pattern that EXACTLY matches RhythmFirstStrategy grammar"""
    path = []
    
    # InitialAssessment (MANDATORY)
    path.extend(['II-Rhythm', 'II-Rate', 'I-Axis', 'aVF-Axis'])
    
    # ExtendedRhythmPhase
    if random.random() > 0.5:
        path.append('V1-Rhythm')
    else:
        path.extend(['V1-Rhythm', 'II-Rhythm'])
    
    # ComponentPhase: PWaveCheck PRCheck QRSCheck STCheck TWaveCheck QTCheck
    # PWaveCheck
    if random.random() > 0.5:
        path.append('II-P')
    else:
        path.extend(['II-P', 'V1-P'])
    
    # PRCheck (MANDATORY in grammar)
    path.append('II-PR')
    
    # QRSCheck: 3 or 4 QRS leads
    qrs_count = random.choice([3, 4])
    qrs_leads = random.sample(['II-QRS', 'V1-QRS', 'V2-QRS', 'V3-QRS', 'V4-QRS', 'V5-QRS'], qrs_count)
    path.extend(qrs_leads)
    
    # STCheck: 2 or 3 ST leads
    st_count = random.choice([2, 3])
    st_leads = random.sample(['II-ST', 'V3-ST', 'V4-ST', 'V2-ST', 'V5-ST'], st_count)
    path.extend(st_leads)
    
    # TWaveCheck: 2 or 3 T leads
    t_count = random.choice([2, 3])
    t_leads = random.sample(['II-T', 'V3-T', 'V4-T', 'V5-T'], t_count)
    path.extend(t_leads)
    
    # QTCheck
    path.append('II-QT')
    
    # RegionalSweep: 1-8 plain leads
    sweep_count = random.randint(1, 8)
    sweep_leads = random.sample(ALL_LEADS, sweep_count)
    path.extend(sweep_leads)
    
    return path

def generate_expert_morphology_first():
    """Generate pattern that EXACTLY matches MorphologyFirstStrategy grammar"""
    path = []
    
    # InitialAssessment
    path.extend(['II-Rhythm', 'II-Rate', 'I-Axis', 'aVF-Axis'])
    
    # BriefRhythm
    path.append('V1-Rhythm')
    
    # DeepMorphologyPhase: ExtendedPWave ExtendedPR ExtendedQRS ExtendedST ExtendedT ExtendedQT
    
    # ExtendedPWave: 3 or 4 P leads
    p_count = random.choice([3, 4])
    p_leads = random.sample(['II-P', 'V1-P', 'I-P', 'III-P'], p_count)
    path.extend(p_leads)
    
    # ExtendedPR: 2 or 3 PR leads
    pr_count = random.choice([2, 3])
    pr_leads = random.sample(['II-PR', 'V1-PR', 'I-PR'], pr_count)
    path.extend(pr_leads)
    
    # ExtendedQRS: 5 or 6 QRS leads
    qrs_count = random.choice([5, 6])
    qrs_leads = random.sample(['II-QRS', 'V1-QRS', 'V2-QRS', 'V3-QRS', 'V4-QRS', 'V5-QRS', 'V6-QRS'], qrs_count)
    path.extend(qrs_leads)
    
    # ExtendedST: 4 or 5 ST leads
    st_count = random.choice([4, 5])
    st_leads = random.sample(['II-ST', 'V2-ST', 'V3-ST', 'V4-ST', 'V5-ST', 'III-ST'], st_count)
    path.extend(st_leads)
    
    # ExtendedT: 4 or 5 T leads
    t_count = random.choice([4, 5])
    t_leads = random.sample(['II-T', 'V3-T', 'V4-T', 'V5-T', 'III-T'], t_count)
    path.extend(t_leads)
    
    # ExtendedQT: 2 QT leads
    path.extend(['II-QT', 'V3-QT'])
    
    # RegionalSweep
    sweep_count = random.randint(1, 8)
    sweep_leads = random.sample(ALL_LEADS, sweep_count)
    path.extend(sweep_leads)
    
    return path

def generate_expert_regional_first():
    """Generate pattern that EXACTLY matches RegionalFirstStrategy grammar"""
    path = []
    
    # InitialAssessment
    path.extend(['II-Rhythm', 'II-Rate', 'I-Axis', 'aVF-Axis'])
    
    # QuickMorphCheck
    path.extend(['II-P', 'V1-P', 'II-QRS', 'V1-QRS'])
    
    # SystematicRegionalExam: 3 or 4 RegionalExam
    num_regions = random.choice([3, 4])
    
    region_patterns = [
        # InferiorExam options
        ['II-QRS', 'II-ST', 'II-T'],
        ['III-QRS', 'III-ST', 'III-T'],
        ['aVF-QRS', 'aVF-ST', 'aVF-T'],
        ['II-QRS', 'II-ST'],
        # LateralExam options
        ['V5-QRS', 'V5-ST', 'V5-T'],
        ['V6-QRS', 'V6-ST', 'V6-T'],
        ['I-QRS', 'I-ST'],
        # AnteriorExam options
        ['V3-QRS', 'V3-ST', 'V3-T'],
        ['V4-QRS', 'V4-ST', 'V4-T'],
        ['V3-QRS', 'V3-ST'],
        # SeptalExam options
        ['V1-QRS', 'V1-ST', 'V1-T'],
        ['V2-QRS', 'V2-ST', 'V2-T'],
        ['V1-QRS', 'V1-ST'],
    ]
    
    selected_regions = random.sample(region_patterns, num_regions)
    for region in selected_regions:
        path.extend(region)
    
    return path

def generate_novice():
    """Generate truly random pattern that should NOT parse"""
    all_possible = [f'{lead}-{comp}' for lead in ALL_LEADS 
                    for comp in ['P', 'QRS', 'ST', 'T']]
    
    length = random.randint(12, 25)
    path = random.sample(all_possible, length)
    
    # Remove systematic starts (novices skip this)
    path = [p for p in path if p not in ['II-Rhythm', 'II-Rate', 'I-Axis', 'aVF-Axis']]
    
    random.shuffle(path)
    
    return path

def generate_intermediate():
    """Generate partially systematic pattern - MUST start correctly to parse!"""
    path = []
    
    # Intermediates DO start systematically (that's why they're not complete novices)
    # But they have incomplete coverage
    path.extend(['II-Rhythm', 'II-Rate', 'I-Axis', 'aVF-Axis'])
    
    # They check SOME components but not all
    # Let's make them follow rhythm-first but incomplete
    
    # Maybe add extended rhythm (50% chance)
    if random.random() > 0.5:
        path.append('V1-Rhythm')
    
    # Check only 1-2 components (not all 6 like experts)
    components_to_check = random.sample(['P', 'QRS', 'ST'], random.randint(1, 2))
    
    for comp in components_to_check:
        if comp == 'P':
            path.append('II-P')  # Just one P check
        elif comp == 'QRS':
            # Only 2 QRS checks (experts do 3-4)
            qrs_leads = random.sample(['II-QRS', 'V1-QRS', 'V3-QRS'], 2)
            path.extend(qrs_leads)
        elif comp == 'ST':
            # Only 2 ST checks
            st_leads = random.sample(['II-ST', 'V3-ST'], 2)
            path.extend(st_leads)
    
    # Maybe add a couple plain leads at end
    if random.random() > 0.5:
        path.extend(random.sample(ALL_LEADS, 2))
    
    # This should parse but with shallow depth (3-4 levels)
    return path

def generate_dataset():
    dataset = []
    sid = 1
    
    print("Generating expert scanpaths...")
    for _ in range(34):
        path = generate_expert_rhythm_first()
        dataset.append({
            'scanpath_id': f'SCAN_EXP_{sid:03d}',
            'expertise_level': 'expert',
            'strategy_type': 'rhythm_first',
            'fixation_sequence': ' '.join(path),
            'sequence_length': len(path)
        })
        sid += 1
    
    for _ in range(33):
        path = generate_expert_morphology_first()
        dataset.append({
            'scanpath_id': f'SCAN_EXP_{sid:03d}',
            'expertise_level': 'expert',
            'strategy_type': 'morphology_first',
            'fixation_sequence': ' '.join(path),
            'sequence_length': len(path)
        })
        sid += 1
    
    for _ in range(33):
        path = generate_expert_regional_first()
        dataset.append({
            'scanpath_id': f'SCAN_EXP_{sid:03d}',
            'expertise_level': 'expert',
            'strategy_type': 'regional_first',
            'fixation_sequence': ' '.join(path),
            'sequence_length': len(path)
        })
        sid += 1
    
    print(f"Generated 100 expert scanpaths")
    
    print("Generating intermediate scanpaths...")
    for _ in range(50):
        path = generate_intermediate()
        dataset.append({
            'scanpath_id': f'SCAN_INT_{sid:03d}',
            'expertise_level': 'intermediate',
            'strategy_type': 'partial_systematic',
            'fixation_sequence': ' '.join(path),
            'sequence_length': len(path)
        })
        sid += 1
    
    print(f"Generated 50 intermediate scanpaths")
    
    print("Generating novice scanpaths...")
    for _ in range(100):
        path = generate_novice()
        dataset.append({
            'scanpath_id': f'SCAN_NOV_{sid:03d}',
            'expertise_level': 'novice',
            'strategy_type': 'random',
            'fixation_sequence': ' '.join(path),
            'sequence_length': len(path)
        })
        sid += 1
    
    print(f"Generated 100 novice scanpaths")
    print(f"\nTotal: {len(dataset)} scanpaths")
    
    return dataset

if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)
    
    dataset = generate_dataset()
    
    filepath = 'data/processed/all_scanpaths.csv'
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['scanpath_id', 'expertise_level', 'strategy_type', 
                      'fixation_sequence', 'sequence_length']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    
    print(f"\nDataset saved to {filepath}")
    print("Done!")