"""Update submission links only after a verified public Hub publication."""
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

root = Path(__file__).resolve().parents[1]
publication = json.loads((root/'submission/HF_PUBLICATION.json').read_text(encoding='utf-8'))
assert publication['public_verified']
url = publication['url']
report_path = root/'submission/REPORT.md'
report = report_path.read_text(encoding='utf-8-sig')
report = report.replace('Không có bằng chứng NB6, dataset riêng, đối chứng reasoning mask, rank sweep hoặc link HuggingFace nên không nhận điểm thưởng đó.',
    'Chưa có bằng chứng NB6, dataset riêng, đối chứng reasoning mask hoặc rank sweep nên không nhận điểm thưởng B1–B4. Adapter đã công khai trên Hugging Face; đề nghị xét B5 theo liên kết bên dưới.')
if '## 9. Adapter công khai' not in report:
    report += f'\n## 9. Adapter công khai — B5\n\n[Hugging Face Hub]({url}) · [Commit xuất bản adapter]({url}/commit/{publication["commit"]}).\n\nĐã xác minh repo truy cập công khai và có adapter_model.safetensors, adapter_config.json, tokenizer, model card cùng kết quả đánh giá. Model card công bố rõ verdict FAILED và giới hạn dữ liệu. Việc chia sẻ adapter không thay đổi phán quyết chất lượng. Theo rubric, đây là bằng chứng để xét điểm thưởng B5 (+2).\n'
report_path.write_text(report, encoding='utf-8')
mapping = root/'submission/EVIDENCE_MAP.md'
mapping.write_text(mapping.read_text(encoding='utf-8').replace(
    'Không yêu cầu điểm thưởng B1–B5 vì chưa có thí nghiệm hoặc công bố tương ứng.',
    f'Đề nghị xét B5 (+2): [adapter công khai]({url}), bằng chứng tại HF_PUBLICATION.json. Chưa yêu cầu B1–B4.'), encoding='utf-8')
(root/'LINKS.md').write_text(f'''# Liên kết nộp Lab 21

- Adapter công khai: {url}
- Repo bài nộp: https://github.com/MinhNhatUet/Day21-Track3-Finetuning-Lab
- Báo cáo: {url}/blob/main/submission/REPORT.md
- Kết quả: {url}/tree/main/results
- Repo bài lab gốc (không phải repo cá nhân): https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-Lab

Phương án B: repo GitHub chứa báo cáo và results; adapter được lưu công khai trên Hugging Face.
''', encoding='utf-8')
files = ['submission/REPORT.md', 'submission/EVIDENCE_MAP.md', 'submission/HF_PUBLICATION.json', 'LINKS.md']
api = HfApi()
commit = api.create_commit(repo_id=url.split('huggingface.co/')[1], repo_type='model',
    operations=[CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(root/f)) for f in files],
    commit_message='Add verified publication links and B5 evidence to report')
print('Updated public submission documents:', commit.oid)
