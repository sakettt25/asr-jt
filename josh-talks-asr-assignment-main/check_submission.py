import json
import os

checklist = {}

# Q1
q1_report = 'artifacts/q1/report.json'
if os.path.exists(q1_report):
    with open(q1_report) as f:
        q1 = json.load(f)
        checklist['Q1'] = {
            'report_exists': True,
            'has_preprocessing': 'preprocessing' in q1,
            'has_baseline_wer': 'results' in q1 and len(q1.get('results', [])) > 0,
            'has_error_taxonomy': 'error_taxonomy' in q1,
            'has_top_fixes': 'top_fixes' in q1,
            'has_implementation': 'implementation_results' in q1,
        }

# Q2
q2_report = 'artifacts/q2/report.json'
if os.path.exists(q2_report):
    with open(q2_report) as f:
        q2 = json.load(f)
        checklist['Q2'] = {
            'report_exists': True,
            'has_number_normalization': 'number_normalization' in q2,
            'has_english_detection': 'english_word_detection' in q2,
        }

# Q3
q3_report = 'artifacts/q3/report.json'
if os.path.exists(q3_report):
    with open(q3_report) as f:
        q3 = json.load(f)
        checklist['Q3'] = {
            'report_exists': True,
            'has_approach': 'approach' in q3,
            'has_classification': 'words_classification' in q3,
            'has_low_confidence_review': 'low_confidence_review' in q3,
            'has_unreliable_categories': 'unreliable_categories' in q3,
        }

# Q4
q4_report = 'artifacts/q4/report.json'
if os.path.exists(q4_report):
    with open(q4_report) as f:
        q4 = json.load(f)
        checklist['Q4'] = {
            'report_exists': True,
            'has_pseudocode': 'pseudocode' in q4,
            'has_models': 'models' in q4,
            'has_results': 'results' in q4,
        }

print('\n' + '='*50)
print('SUBMISSION COMPLETENESS CHECKLIST')
print('='*50 + '\n')

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    if q in checklist:
        print(f'✅ {q}: Present')
        for key, val in checklist[q].items():
            if key != 'report_exists':
                status = '✓' if val else '✗'
                print(f'    {status} {key}')
    else:
        print(f'❌ {q}: Missing')

print('\n' + '='*50)
