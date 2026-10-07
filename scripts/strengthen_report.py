"""Extract real training logs and add evidence to the report; never rerun training."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'colab/Lab21_RUN_ALL.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))
text = '\n'.join(''.join(o.get('text', [])) for c in nb['cells'] for o in c.get('outputs', []))
curves = {k: [] for k in ('correct', 'attn_only', 'wrong_lr', 'qlora')}
run = 'correct'
for line in text.splitlines():
    if line.startswith('RUN '):
        run = line.split()[1].rstrip(':')
    if line.startswith("{'loss':"):
        row = ast.literal_eval(line)
        curves[run].append(row)
assert all(len(rows) == 6 for rows in curves.values())
evidence = {
    'source': 'colab/Lab21_RUN_ALL.ipynb saved training outputs',
    'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'notes': 'Values preserved as printed strings. Epochs are printed positions, not inferred optimizer steps.',
    'curves': curves,
    'pretraining_baseline': json.loads((ROOT / 'submission/evidence_before_recovery/baselines_frozen.json').read_text(encoding='utf-8')),
    'observed_training_setup': {'gpu': 'Tesla T4', 'gpu_memory_gb_printed': 14.6,
        'max_length': 1024, 'train': 225, 'validation': 25, 'epochs': 2.0,
        'per_device_batch': 1, 'gradient_accumulation': 16, 'max_steps': 30,
        'packing': False, 'padding_free': False},
}
(ROOT / 'results/training_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
p = ROOT / 'submission/REPORT.md'
report = p.read_text(encoding='utf-8-sig')
length = '''
### Giải thích lựa chọn độ dài bằng số đo

Theo thống kê, ngay cả mẫu dài nhất (101 token) cũng ngắn hơn cả 256 lẫn 1024. Trong `build_example`, max_length là ngưỡng cắt, không phải lệnh đệm mọi mẫu lên đúng ngưỡng; mã chỉ cắt khi chuỗi dài hơn giới hạn. Vì vậy, trên 250 mẫu đã đo, riêng bước cắt token cho cùng kết quả ở hai giới hạn này. Batch train bằng 1 và packing tắt theo log; không có cơ sở để nói dùng 1024 khiến số token thực tế tăng bốn lần. Điều này giải thích vì sao cấu hình hiện tại không làm mất câu trả lời, nhưng không biến lựa chọn mặc định thành một thí nghiệm tối ưu bộ nhớ. Khuyến nghị dựa trên p95 vẫn là 256; không nhận đã đo tốc độ hoặc VRAM ở giới hạn đó. Nguồn: token_stats.json, training_evidence.json và src/labkit/data.py.

'''
timeline = '''
### Bằng chứng baseline có trước huấn luyện

Output notebook ghi đúng trình tự NB1 → NB2 → NB3 → NB4 → NB5, rồi mới chạy lại NB2+NB5 đầy đủ. Bản JSON trước train được giữ trong `results/training_evidence.json`, mục `pretraining_baseline`.

| Giai đoạn | Target / regression được chấm | Target (a) | Target (b) | Vai trò |
|---|---|---:|---:|---|
| NB2 trước train | 8 / 8 | 0.0000 | 0.6875 | Xác nhận prompt (b) mạnh hơn (a) trước khi học adapter |
| NB2 sau train | 50 / 15 | 0.0000 | 0.7650 | Mở rộng phép đo với cùng prompt và corpus |

Cả hai lưu cùng SHA prompt `719e74d3b6232053`. Như vậy có bằng chứng baseline mạnh đã được đo trước train, thay vì chọn prompt sau khi thấy adapter. Tuy nhiên các điểm trên tập đầy đủ chỉ được đo sau train; cần giữ rõ khác biệt này khi chấm tiêu chí 3.1. Không thể suy ra hiệu năng từng mẫu từ các trung bình trong bảng.

'''
controls = '''
### Biến thay đổi và biến kiểm soát

| Đối chứng với correct | Biến nghiên cứu | Điều chỉnh để phép so công bằng | Giữ cố định |
|---|---|---|---|
| attn_only | Vị trí adapter: text-linear → q,v | r: 16 → 283; alpha: 32 → 566 để gần khớp số tham số và giữ alpha/r=2 | Base, LR, precision, dữ liệu, ngân sách 30 step |
| wrong_lr | LR: 0.0001 → 0.00001 | Không đổi rank/alpha | Base, vị trí, precision, dữ liệu, ngân sách 30 step |
| qlora | Base 16-bit → base 4-bit | Eval cũng nạp base 4-bit | Base model ID, vị trí, rank/alpha, LR, dữ liệu, ngân sách 30 step |

Với attention-only, giữ rank 16 sẽ chỉ có 1,835,008 tham số theo log, nên không thể dùng cách đó để tách tác động vị trí khỏi ngân sách. Điều chỉnh rank/alpha là biện pháp kiểm soát có chủ đích, nhưng không tách hoàn toàn mọi ảnh hưởng biểu diễn của rank. Kết luận đúng phạm vi là hai cấu hình gần cùng ngân sách hòa trên target, không phải vị trí hay rank đều vô tác dụng.

**Xếp hạng theo target:** correct = attn_only (0.97) > qlora (0.94) > wrong_lr (0.00). **Theo training loss tăng dần:** attn_only < correct < qlora < wrong_lr. Sự khác biệt ở hai vị trí đầu minh họa vì sao không lấy training loss làm tiêu chí thắng tác vụ.

### Diễn biến loss thực tế, không suy đoán từ một số cuối

Nguồn: `results/training_evidence.json`, trích nguyên các mốc log Colab. Epoch là giá trị in trong log; loss ở mỗi mốc là giá trị logging của trainer, không phải loss validation.

| Epoch | correct | attn_only | wrong_lr | qlora |
|---|---:|---:|---:|---:|
'''
for i in range(6):
    controls += '| ' + curves['correct'][i]['epoch'] + ' | ' + ' | '.join(curves[k][i]['loss'] for k in curves) + ' |\n'
controls += '''
Wrong_lr không phẳng hoàn toàn: loss giảm từ 2.163 xuống 1.119, nhưng chậm và còn cao so với correct giảm từ 2.163 xuống 0.02474. Nếu chỉ thấy loss đang giảm, có thể kết luận nhầm run đã học được tác vụ; target và format bằng 0 bác bỏ kết luận đó ở ngân sách hiện tại. Ngược lại, loss correct rất thấp không ngăn được regression trên tập đầy đủ. Một số grad_norm được log là nan; dữ liệu hiện có không đủ xác định số cập nhật bị bỏ qua bởi gradient scaling. Cùng 30 step ghi nhận bảo đảm ngân sách cấu hình, không chứng minh mọi cập nhật hữu hiệu đều giống nhau. Không cần gán nguyên nhân chắc chắn khi chưa có log scaler.

'''
if '### Giải thích lựa chọn độ dài bằng số đo' not in report:
    report = report.replace('## 2. Bằng chứng mask và template', length+'## 2. Bằng chứng mask và template')
    report = report.replace('## 4. Đối chứng cấu hình', timeline+'## 4. Đối chứng cấu hình')
    report = report.replace('## 5. Phán quyết:', controls+'## 5. Phán quyết:')
p.write_text(report, encoding='utf-8')
print('Extracted 24 real loss log entries and strengthened report evidence.')
