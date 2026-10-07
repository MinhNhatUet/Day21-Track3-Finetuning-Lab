# Bản đồ bằng chứng theo rubric

**Ngô Đinh Minh Nhật — 2A202602569**

Tài liệu này giúp đối chiếu báo cáo với artefact; không tự cấp điểm.

| Tiêu chí | Bằng chứng | Vị trí báo cáo / giới hạn |
|---|---|---|
| 1.1 Mask (10) | results/mask_proof.json | Mục 2: hai assert true, 39/94 token |
| 1.2 Template (5) | results/template_check.json | Mục 2: thẻ và nội dung reasoning được giữ |
| 1.3 Độ dài (5) | results/token_stats.json, training_evidence.json | Mục 1: p95=98, đề xuất 256, thực tế 1024; giải thích ngưỡng cắt và giới hạn |
| 1.4 Adapter (10) | adapters/correct/adapter_model.safetensors, adapter_config.json; results/runs.csv | Mục 4: loss, thời gian, VRAM |
| 2.1 Khớp tham số (10) | results/runs.csv | Mục 4: 32,456,704 so với 32,464,896, chênh khoảng 0.0252% |
| 2.2 Cùng step (5) | results/runs.csv | Bốn run đều 30 step; lưu ý ngân sách ghi nhận khác số cập nhật hữu hiệu khi có gradient bất thường |
| 2.3 Một biến nghiên cứu (5) | results/runs.csv, training_evidence.json | Mục 4: bảng biến nghiên cứu và điều chỉnh rank/alpha để kiểm soát ngân sách |
| 2.4 Vị trí và rank (5) | results/autopsy.json, runs.csv | Target hòa dù training loss khác; không tuyên bố rank sweep |
| 3.1 Baseline (5) | results/training_evidence.json: pretraining_baseline; baselines_frozen.json | Baseline 8 mẫu trước train; baseline đầy đủ đo sau train, cùng SHA prompt. Không che giấu khác biệt thời điểm |
| 3.2 Bốn nhóm (10) | results/verdict.json | Mục 3: target, regression, format, latency |
| 3.3 Phán quyết (5) | results/verdict.json | Mục 5: FAILED do regression, phân biệt với lỗi pipeline |
| 3.4 Định tính (5) | results/qualitative.json, data/eval_target.jsonl | Sáu preview, ba ca FT sai urgency. Thiếu baseline từng mẫu, chưa chứng minh hai ca FT thua baseline |
| 4.1 Cấu trúc (5) | submission/REPORT.md | Lựa chọn, lý do, mask, baseline, đối chứng, verdict và bài học |
| 4.2 Kết luận (5) | REPORT.md mục 7 | Kết luận trên 150 từ, quyết định không deploy và lý do |
| 4.3 Nhất quán (5) | results/*.json, runs.csv; recovery_provenance.json | Số khôi phục có nguồn; không phục dựng dự đoán không còn trong log |
| 4.4 Phản tư (5) | REPORT.md mục 7; REFLECTION.md | Bài học cụ thể từ kết quả; trải nghiệm cá nhân cần tác giả xác nhận |

Đề nghị xét B5 (+2): [adapter công khai](https://huggingface.co/nhatnhoem/lab21-qwen35-4b-triage-lora), bằng chứng tại HF_PUBLICATION.json. Chưa yêu cầu B1–B4.
