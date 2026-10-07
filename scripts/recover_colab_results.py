"""Recover only values actually printed by the full-eval Colab run."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / 'colab/Lab21_RUN_ALL.ipynb'
nb = json.loads(nb_path.read_text(encoding='utf-8'))
texts = [''.join(''.join(o.get('text', [])) for o in c.get('outputs', []))
         for c in nb['cells']]
full = next(t for t in reversed(texts) if 'target=50  regression=15' in t)
start = full.index('{\n  "tier":')
baseline, _ = json.JSONDecoder().raw_decode(full[start:])
assert baseline['n_target'] == 50 and baseline['n_regression'] == 15
assert baseline['smoke_mode'] is False
archive = ROOT / 'submission/evidence_before_recovery'
archive.mkdir(exist_ok=True)
for name in ('baselines_frozen.json', 'autopsy.json', 'qualitative.json'):
    dest = archive / name
    if not dest.exists():
        dest.write_bytes((ROOT / 'results' / name).read_bytes())

def write(name, value):
    (ROOT / 'results' / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

write('baselines_frozen.json', baseline)
autopsy = []
qualitative = []
mode = None
for line in full.splitlines():
    if line.startswith('| run | target | format | latency_ms | n |'):
        mode = 'autopsy'
    elif line.startswith('| i | ticket | ft_score | ft_pred |'):
        mode = 'qualitative'
    elif line.startswith('| ') and not line.startswith('|---'):
        cols = [v.strip() for v in line.strip().strip('|').split('|')]
        if mode == 'autopsy' and cols[0] in ('correct', 'attn_only', 'wrong_lr', 'qlora'):
            autopsy.append(dict(run=cols[0], target=float(cols[1]), format=float(cols[2]),
                                latency_ms=float(cols[3]), n=int(cols[4])))
        elif mode == 'qualitative' and cols[0].isdigit():
            qualitative.append(dict(i=int(cols[0]), ticket=cols[1], ft_score=float(cols[2]),
                                    ft_pred='|'.join(cols[3:]),
                                    source='full-eval notebook printed preview; not complete prediction'))
assert len(autopsy) == 4 and all(r['n'] == 50 for r in autopsy)
assert len(qualitative) == 6
write('autopsy.json', autopsy)
write('qualitative.json', qualitative)
write('recovery_provenance.json', {
    'source': 'colab/Lab21_RUN_ALL.ipynb',
    'method': 'Exact JSON and Markdown table extraction from saved full-eval output',
    'baseline': 'Complete JSON recovered without changing values',
    'autopsy': 'All four full-eval rows recovered at printed precision',
    'qualitative': 'Only six printed previews recovered; original 50-row file unavailable',
    'missing': ['44 qualitative rows', 'complete FT predictions', 'per-item baseline predictions'],
    'old_artifacts': 'submission/evidence_before_recovery/',
    'chronology': '8-item baseline before training; full baseline and evaluation rerun after training',
})
print('Recovered full baseline, four autopsy rows, six qualitative previews from notebook output.')
