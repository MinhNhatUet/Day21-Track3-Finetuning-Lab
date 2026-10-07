"""Package Option A, including provenance for recovered results."""
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
prefix = 'lab21_2A202602569/'
files = [p for folder in ('submission', 'results', 'notebooks')
         for p in (root / folder).rglob('*')
         if p.is_file() and '__pycache__' not in p.parts and p.name != '.gitkeep']
files += [root / 'adapters/correct' / name
          for name in ('adapter_config.json', 'adapter_model.safetensors')]
files += [root / name for name in (
    'requirements.txt', 'scripts/recover_colab_results.py', 'scripts/strengthen_report.py',
    'scripts/package_submission.py', 'data/eval_target.jsonl', 'data/checksums.json')]
source = root / 'colab/Lab21_RUN_ALL.ipynb'
notebook = json.loads(source.read_text(encoding='utf-8'))
logs = '\n\n'.join(
    f'CELL {i}\n' + ''.join(''.join(o.get('text', [])) for o in c.get('outputs', []))
    for i, c in enumerate(notebook['cells']) if c.get('outputs'))
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
archive = root / 'lab21_2A202602569.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files:
        z.write(p, prefix + p.relative_to(root).as_posix())
    # Code notebooks are output-free; historical logs remain separately reviewable.
    z.writestr(prefix + 'colab/Lab21_RUN_ALL.ipynb', json.dumps(notebook, ensure_ascii=False, indent=2))
    z.writestr(prefix + 'submission/evidence/colab_output.txt', logs)
    z.write(source, prefix + 'submission/evidence/Lab21_RUN_ALL_executed.ipynb')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    report = z.read(prefix + 'submission/REPORT.md').decode('utf-8-sig')
    assert 'Diễn biến loss thực tế' in report
    evidence = json.loads(z.read(prefix + 'results/training_evidence.json'))
    assert all(len(rows) == 6 for rows in evidence['curves'].values())
print(f'ZIP checked: {archive.name}; {archive.stat().st_size / 1024**2:.1f} MiB')
