"""Summarize labeled sample inference; never treat failed inference as a prediction."""
import argparse
import json
import math
from pathlib import Path

LABELS = {
    'face': ('affected', 'unaffected'),
    'speech': ('dysarthria', 'healthy'),
    'movement': ('Incomplete', 'Complete'),
}


def summarize(rows):
    summaries = []
    for name, (positive, negative) in LABELS.items():
        selected = [r for r in rows if r.get('modality') == name]
        valid = []
        for row in selected:
            score = row.get('score')
            if (row.get('status') == 'success'
                    and isinstance(score, (int, float))
                    and math.isfinite(score) and 0 <= score <= 1
                    and row.get('expected') in (positive, negative)):
                valid.append(row)
        correct = sum((positive if r['score'] >= 0.5 else negative) == r['expected'] for r in valid)
        required_split = {'speech': 'test speaker per repository split', 'movement': 'dataset Test folder'}.get(name)
        is_test = bool(selected) and required_split is not None and all(r.get('split') == required_split for r in selected)
        summaries.append({
            'modality': name,
            'metric': 'Test Accuracy (subset)' if is_test else 'Sample Accuracy (split unverified)',
            'selected': len(selected), 'evaluated': len(valid),
            'failed_or_unscorable': len(selected) - len(valid),
            'correct': correct,
            'accuracy_percent': 100 * correct / len(valid) if valid else None,
            'threshold': 0.5,
        })
    return summaries


def write_summary(rows, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    lines = ['# Accuracy summary', '',
             'Accuracy = correct predictions / successfully evaluated labeled samples, at threshold 0.5.',
             'Failed or unscorable inputs are excluded from the denominator and shown separately.',
             'These are selected samples, not the entire test set. Face training overlap is unknown.', '',
             '| Module | Metric | Accuracy | Correct/evaluated | Selected | Failed/unscorable |',
             '|---|---|---:|---:|---:|---:|']
    print('\nACCURACY SUMMARY (threshold = 0.5)', flush=True)
    for item in summary:
        accuracy = 'N/A' if item['accuracy_percent'] is None else f"{item['accuracy_percent']:.2f}%"
        counts = f"{item['correct']}/{item['evaluated']}"
        print(f"{item['modality']:10s} | {item['metric']}: {accuracy} ({counts}) | selected={item['selected']} | failed/unscorable={item['failed_or_unscorable']}", flush=True)
        lines.append(f"| {item['modality']} | {item['metric']} | {accuracy} | {counts} | {item['selected']} | {item['failed_or_unscorable']} |")
    print('Subset only; face split is unverified. These values are not clinical validation.', flush=True)
    (output / 'accuracy_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'ACCURACY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate accuracy from saved inference results without rerunning models.')
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/data_sample_test/results.json')
    args = parser.parse_args()
    write_summary(json.loads(args.results.read_text(encoding='utf-8')), args.results.parent)
