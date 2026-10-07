# Lab 21 — LoRA cho ticket CSKH tiếng Việt

**Họ tên:** Ngô Đinh Minh Nhật · **MSSV:** 2A202602569  
**Ngày:** 07/10/2026

> Kết quả chính: đánh giá đủ 50 target và 15 regression. Model FAILED cổng hồi quy vì năng lực tổng quát giảm quá ngưỡng. Một phần artefact được khôi phục từ output notebook do bản sao Drive chưa cập nhật; xem mục 8.

## 1. Lựa chọn và thiết kế

Model unsloth/Qwen3.5-4B, tier T4, GPU Tesla T4 (14.6 GB theo log), precision fp16.

**Vì sao chọn model này.** Colab free chỉ có T4 16 GB và T4 không hỗ trợ bf16, nên cần một base đủ nhỏ để LoRA 16-bit chạy fp16 mà vẫn còn chỗ cho đối chứng QLoRA. Qwen3.5-4B vừa khít: peak VRAM của run correct là 8.78 GB, còn dư khoảng 6 GB. Model cũng xử lý tốt tiếng Việt và có chế độ thinking, nên kiểm tra template `<think>` (mục 2) là bài kiểm tra thực tế. Base được giữ nguyên giữa baseline và adapter để mọi chênh lệch đến từ LoRA.

**Vì sao chọn dataset này.** 250 ticket CSKH tiếng Việt, nhãn JSON gồm bốn trường intent, urgency, product, sentiment. Đây là kiểu tác vụ fine-tune có lợi rõ nhất: định dạng đầu ra cố định và tập nhãn đóng mà prompt khó ép hoàn toàn. Nhãn khách quan nên chấm được theo từng trường, không cần LLM judge, và phép so baseline (b) với LoRA không bị nhiễu bởi người chấm.

Log NB1 xác nhận 225 train / 25 validation, seed 42. NB3: hai epoch, batch 1, gradient accumulation 16 (batch hiệu dụng 16 < 32), 30 optimizer step; bốn run đều dùng 30 step.

**max_length.** Thống kê token: p95=98, p99=100, max=101, đề xuất max_length=256. Lần chạy này vẫn dùng 1024 theo tier mặc định, chưa đặt theo p95. Vì `build_example` chỉ cắt chứ không đệm, và mẫu dài nhất (101) ngắn hơn cả hai ngưỡng, nên 1024 không làm mất token nào nhưng cũng không được chọn dựa trên số đo. Lần chạy sau nên đặt 256.

**Thứ tự đo baseline.** Baseline được đo trên tám mẫu trước train; tập đầy đủ 50/15 được chạy lại sau train với cùng SHA prompt và checksum eval (chi tiết ở mục 3).

## 2. Bằng chứng mask và template

Nguồn: results/mask_proof.json, template_check.json và output NB1.

| Kiểm tra | Giá trị |
|---|---:|
| Mask | assistant-only |
| Token supervised / tổng | 39 / 94 |
| supervised_fraction | 0.4149 |
| answer_is_supervised | true |
| question_is_masked | true |
| Template giữ thẻ mở và nội dung reasoning mẫu | true / true |

Phần được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Câu hỏi được mask, JSON trả lời nằm trong loss. Preview cho thấy phần think rỗng đầu lượt assistant được mask, còn thẻ đóng được tính loss. Phép kiểm tra template riêng xác nhận nội dung reasoning mẫu được giữ; điều đó không bảo đảm model sau train sẽ sinh reasoning hữu ích.

## 3. Ba baseline trên tập đầy đủ

Nguồn: results/baselines_frozen.json, verdict.json và output NB2/NB5. SHA prompt tối ưu: 719e74d3b6232053.

| Run | target | regression | format | latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3731.2 |
| (b) Base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1093.2 |
| (c) LoRA correct | 0.9700 | 0.6556 | 1.0000 | 1615.3 |

Target là độ chính xác trung bình theo trường, không phải tỷ lệ ticket đúng hoàn toàn. Regression là keyword recall trung bình trên 15 câu. Format dùng bộ trích JSON linh hoạt và kiểm tra khóa, không chứng minh output luôn là JSON thuần không kèm văn bản. Prompt tối ưu mạnh hơn naive: target tăng 0.765, format lên 1. Naive có format bằng 0 nên không thể coi target bằng 0 là bằng chứng model không hiểu nội dung.

LoRA tăng 20.5 điểm phần trăm target so với baseline tối ưu, nhưng regression giảm khoảng 13.56 điểm phần trăm. Latency tăng 522.1 ms/mẫu, khoảng 47.8%. Không có bằng chứng prompt bị sửa để làm yếu baseline.


### Bằng chứng baseline có trước huấn luyện

Output notebook ghi đúng trình tự NB1 → NB2 → NB3 → NB4 → NB5, rồi mới chạy lại NB2+NB5 đầy đủ. Bản JSON trước train được giữ trong `results/training_evidence.json`, mục `pretraining_baseline`.

| Giai đoạn | Target / regression được chấm | Target (a) | Target (b) | Vai trò |
|---|---|---:|---:|---|
| NB2 trước train | 8 / 8 | 0.0000 | 0.6875 | Xác nhận prompt (b) mạnh hơn (a) trước khi học adapter |
| NB2 sau train | 50 / 15 | 0.0000 | 0.7650 | Mở rộng phép đo với cùng prompt và corpus |

Cả hai lưu cùng SHA prompt `719e74d3b6232053`. Như vậy có bằng chứng baseline mạnh đã được đo trước train, thay vì chọn prompt sau khi thấy adapter. Tuy nhiên các điểm trên tập đầy đủ chỉ được đo sau train; cần giữ rõ khác biệt này khi chấm tiêu chí 3.1. Không thể suy ra hiệu năng từng mẫu từ các trung bình trong bảng.

## 4. Đối chứng cấu hình

Nguồn: results/runs.csv và autopsy.json. Target của cả bốn run chấm trên 50 mẫu.

| Run | Vị trí | r | Tham số trainable | LR | Training loss | Target | Train (s) | Peak VRAM (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 | 32,464,896 | 0.0001 | 0.6265 | 0.97 | 397.4 | 8.78 |
| attn_only | q,v | 283 | 32,456,704 | 0.0001 | 0.5374 | 0.97 | 273.8 | 8.79 |
| wrong_lr | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.00 | 406.5 | 8.78 |
| qlora | text-linear, base 4-bit | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.94 | 475.0 | 3.86 |

**4.1. Vị trí và rank.** Attention-only ít hơn 8,192 tham số, sai lệch khoảng 0.0252%, dưới ngưỡng 5%. Rank 283 là điều chỉnh để khớp ngân sách khi đổi vị trí, không phải quét rank độc lập. Hai run hòa target 0.97 dù attention-only có loss thấp hơn. Thứ tự theo loss phân biệt hai cấu hình nhưng thứ tự theo target không phân biệt, nên không có bằng chứng text-linear thắng q,v ở đây. Attention-only có latency 996.9 ms/mẫu, thấp hơn correct 1615.3; chưa có regression của đối chứng này để kết luận đủ điều kiện thay thế correct. Cần nhiều seed để đánh giá độ ổn định.

**4.2. Learning rate.** Wrong_lr giảm LR mười lần, giữ rank, vị trí, precision và số step. Training loss 1.5702 cao hơn correct 0.6265, target và format bằng 0. Log correct có các mốc loss giảm từ 2.163 xuống 0.02474, nhưng có dao động cuối và một số grad_norm=nan; không khẳng định mọi bước đều ổn định. Cột final_loss được gán từ result.training_loss, không phải riêng loss step cuối. LR thấp phù hợp giả thuyết adapter chưa học đủ trong ngân sách hiện tại; bỏ qua LR có thể khiến ta kết luận nhầm model hoặc LoRA không phù hợp. Lỗi format cũng ảnh hưởng trực tiếp target nên chưa tách được hoàn toàn lỗi ngữ nghĩa.

**4.3. QLoRA.** Peak VRAM giảm 4.92 GB, khoảng 56.0%, nhưng train tăng 77.6 giây, khoảng 19.5%. Target giảm 0.03 và latency tăng từ 1615.3 lên 1948.0 ms/mẫu; format vẫn bằng 1. NB5 nạp base 4-bit cho adapter QLoRA để khớp cấu hình train. Kết quả ủng hộ ưu tiên LoRA 16-bit khi đủ VRAM trên tác vụ này, nhưng chưa chứng minh QLoRA luôn không phù hợp. Tiết kiệm bộ nhớ vẫn hữu ích khi GPU hạn chế; thí nghiệm chỉ có một seed và chưa đo regression của đối chứng.


### Biến thay đổi và biến kiểm soát

| Đối chứng với correct | Biến nghiên cứu | Điều chỉnh để phép so công bằng | Giữ cố định |
|---|---|---|---|
| attn_only | Vị trí adapter: text-linear → q,v | r: 16 → 283; alpha: 32 → 566 để gần khớp số tham số và giữ alpha/r=2 | Base, LR, precision, dữ liệu, ngân sách 30 step |
| wrong_lr | LR: 0.0001 → 0.00001 | Không đổi rank/alpha | Base, vị trí, precision, dữ liệu, ngân sách 30 step |
| qlora | Base 16-bit → base 4-bit | Eval cũng nạp base 4-bit | Base model ID, vị trí, rank/alpha, LR, dữ liệu, ngân sách 30 step |

Với attention-only, giữ rank 16 sẽ chỉ có 1,835,008 tham số theo log, nên không thể dùng cách đó để tách tác động vị trí khỏi ngân sách. Điều chỉnh rank/alpha là biện pháp kiểm soát có chủ đích, nhưng không tách hoàn toàn mọi ảnh hưởng biểu diễn của rank. Kết luận đúng phạm vi là hai cấu hình gần cùng ngân sách hòa trên target, không phải vị trí hay rank đều vô tác dụng.

**Vậy đâu là đòn bẩy?** Ở ngân sách 30 step trên tác vụ này, **learning rate** là đòn bẩy lớn nhất: chỉ đổi LR ×0.1 làm target rơi từ 0.97 xuống 0.00. Khi đã khớp tổng số tham số, **vị trí** (text-linear hay q,v) không tạo khác biệt về target (0.97 = 0.97). Vì vậy tăng rank để bù vị trí không mang lại gì thêm; rank chỉ đóng vai trò cân ngân sách. Muốn biết rank có là đòn bẩy riêng hay không cần quét rank ở vị trí cố định (B4), thí nghiệm này chưa làm.

**Xếp hạng theo target:** correct = attn_only (0.97) > qlora (0.94) > wrong_lr (0.00). **Theo training loss tăng dần:** attn_only < correct < qlora < wrong_lr. Sự khác biệt ở hai vị trí đầu minh họa vì sao không lấy training loss làm tiêu chí thắng tác vụ.

### Diễn biến loss thực tế, không suy đoán từ một số cuối

Nguồn: `results/training_evidence.json`, trích nguyên các mốc log Colab. Epoch là giá trị in trong log; loss ở mỗi mốc là giá trị logging của trainer, không phải loss validation.

| Epoch | correct | attn_only | wrong_lr | qlora |
|---|---:|---:|---:|---:|
| 0.3556 | 2.163 | 2.163 | 2.163 | 2.155 |
| 0.7111 | 1.384 | 0.8236 | 2.066 | 1.731 |
| 1 | 0.1407 | 0.1482 | 1.606 | 0.2408 |
| 1.356 | 0.02891 | 0.04081 | 1.326 | 0.05115 |
| 1.711 | 0.01714 | 0.02233 | 1.141 | 0.0308 |
| 2 | 0.02474 | 0.02609 | 1.119 | 0.02622 |

Wrong_lr không phẳng hoàn toàn: loss giảm từ 2.163 xuống 1.119, nhưng chậm và còn cao so với correct giảm từ 2.163 xuống 0.02474. Nếu chỉ thấy loss đang giảm, có thể kết luận nhầm run đã học được tác vụ; target và format bằng 0 bác bỏ kết luận đó ở ngân sách hiện tại. Ngược lại, loss correct rất thấp không ngăn được regression trên tập đầy đủ. Một số grad_norm được log là nan; dữ liệu hiện có không đủ xác định số cập nhật bị bỏ qua bởi gradient scaling. Cùng 30 step ghi nhận bảo đảm ngân sách cấu hình, không chứng minh mọi cập nhật hữu hiệu đều giống nhau. Không cần gán nguyên nhân chắc chắn khi chưa có log scaler.

## 5. Phán quyết: FAILED vì hồi quy năng lực tổng quát

**target_delta=+0.205; regression_delta=-0.13555555555555554; valid_trace_rate=0.0.** Ngưỡng suy giảm regression cho phép là 0.02.

LoRA cải thiện target lên 0.97, nhưng regression giảm từ 0.7911 xuống 0.6556. Mức giảm khoảng 0.1356 vượt ngưỡng 0.02 nên cổng trả FAILED là đúng tiêu chí đặt trước. Không sửa ngưỡng chỉ vì target đẹp hơn. Kết quả phù hợp nguy cơ chuyên biệt hóa vào tác vụ và suy giảm ở câu hỏi tổng quát, nhưng keyword recall chưa đủ xác định cơ chế quên hoặc từng lỗi nếu thiếu output đầy đủ. Lần đánh giá tám mẫu trước đó PASSED với regression không đổi; mở rộng tập mới phát hiện hồi quy. Vì vậy smoke test kiểm tra pipeline không thay thế đánh giá cuối. Format vẫn bằng 1 và latency tăng cho thấy cổng FAILED đến từ năng lực ngoài tác vụ, không phải định dạng. Trace rate bằng 0 chưa chứng minh reasoning collapse vì không có thí nghiệm reasoning trước/sau tương ứng. Có thể thử replay dữ liệu tổng quát hoặc điều chỉnh ngân sách trên tập phát triển riêng, rồi chấm lại với cổng cũ; chưa có phép đo nào chứng minh các biện pháp này đã sửa được hồi quy.

## 6. Định tính từ sáu preview của lần đánh giá đầy đủ

Notebook mới in sáu dòng, đã khôi phục đúng các dòng đó; không suy đoán 44 dòng còn thiếu. Nhãn và ticket đầy đủ lấy từ data/eval_target.jsonl. Chỉ số i bắt đầu từ 0. Không có baseline từng mẫu nên không gọi ca FT sai nhãn là ca thua baseline.

| i | Ticket | Nhãn: intent / urgency / product / sentiment | Điểm FT | Nhận xét |
|---|---|---|---:|---|
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | hoan_tien / thap / bình giữ nhiệt / tich_cuc | 0.75 | Sai urgency: trung_binh thay vì thap |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | san_pham_loi / thap / nồi chiên không dầu / trung_tinh | 0.75 | Sai urgency: trung_binh thay vì thap |
| 12 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều. | san_pham_loi / thap / áo khoác gió / tich_cuc | 0.75 | Sai urgency: trung_binh thay vì thap |
| 47 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper không gọi. Hỏi cho biết thôi. Shop hỗ trợ tốt. | van_chuyen / thap / ốp lưng điện thoại / tich_cuc | 1.00 | Điểm ghi nhận đúng 4/4 trường; chỉ còn preview |
| 48 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. Mong shop phản hồi. Nhờ shop kiểm tra. | hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh | 1.00 | Điểm ghi nhận đúng 4/4 trường; chỉ còn preview |
| 49 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm nhé. Shop xem giúp. | san_pham_loi / trung_binh / ốp lưng điện thoại / trung_tinh | 1.00 | Điểm ghi nhận đúng 4/4 trường; chỉ còn preview |

Ba ca i=3,5,12 đều chứa “khi nào tiện” nhưng FT dự đoán urgency trung bình thay vì thấp. Đây là mẫu lỗi cần kiểm tra về độ bao phủ cách diễn đạt trong train; không sửa nhãn eval. Preview bị cắt ở 90 ký tự theo NB5, nên thiếu dấu đóng trong bản in không chứng minh output gốc sai JSON. Các ca điểm 1.0 minh họa xử lý đúng những ticket khác nhưng không chứng minh thắng baseline.

**Giới hạn rubric 3.4:** có sáu ví dụ và ba ca FT còn sai, nhưng chưa đủ bằng chứng cho ít nhất hai ca FT thua baseline. Thiếu output baseline từng mẫu và output FT đầy đủ. Không thể khôi phục những thông tin này từ số trung bình; công khai thiếu sót để người chấm đánh giá đúng.

## 7. Kết luận và bài học

**Kết luận: không deploy adapter này.** Target lên 0.97 nhìn rất đẹp, hơn prompt tối ưu 20.5 điểm, nhưng regression tụt từ 0.7911 xuống 0.6556, gấp hơn sáu lần ngưỡng 0.02 mà em đã chấp nhận trước khi chạy. Nếu bây giờ nới ngưỡng thì cả cổng hồi quy mất ý nghĩa. Lý do em tin đây là quên thảm hoạ chứ không phải lỗi pipeline: format vẫn bằng 1, mask đúng (0.41 token được tính loss), và cả 225 mẫu train đều cùng một dạng prompt phân loại JSON. 30 step với LR 1e-4 trên dữ liệu đơn điệu như vậy đủ để model kéo về phía tác vụ hẹp, nên khi hỏi kiến thức chung thì trả lời kém đi. Latency cũng tăng 522 ms/mẫu so với base + prompt (b). Với bài toán triage ticket, prompt (b) đạt 0.765 mà không mất gì; muốn dùng LoRA thì phải trộn thêm 1–5% dữ liệu tổng quát rồi chấm lại với đúng ngưỡng cũ.

**Điều em học được**

1. **Em suýt nộp một kết luận sai.** Lần chạy đầu, em để trống ô `EVAL_LIMIT` và nghĩ là đang chạy full, nhưng log in ra `eval_limit=8`. Với 8 mẫu, regression trước và sau đều 0.75, verdict PASSED, em đã định viết report luôn. Chạy lại đủ 50/15 (thêm 24 phút) thì verdict thành FAILED. Từ giờ việc đầu tiên em làm là đọc dòng config in ra trong log, không tin vào những gì mình nghĩ là đã đặt.

2. **Loss đang giảm không có nghĩa là model đang học được việc.** Run `wrong_lr` có loss giảm từ 2.163 xuống 1.119, nhìn đồ thị thì tưởng ổn, nhưng target và format đều bằng 0. Ngược lại, `attn_only` có loss thấp hơn `correct` (0.5374 so với 0.6265) mà target vẫn chỉ hòa 0.97. Trước lab em hay nhìn loss để chọn run; giờ em chỉ xếp hạng bằng điểm trên tập eval.

3. **Thứ em tưởng là đòn bẩy lại không phải.** Em đoán gắn LoRA vào toàn bộ linear sẽ thắng hẳn q,v. Khi khớp ngân sách tham số (r=283 cho q,v), hai bên hòa nhau. Chỉ đổi LR mười lần mới làm kết quả sụp. Ở quy mô này, LR quan trọng hơn vị trí rất nhiều.

4. **Bài học về hạ tầng.** T4 không có bf16, nên phải chạy fp16 và log có vài `grad_norm=nan`. QLoRA tiết kiệm 4.92 GB VRAM nhưng chạy lâu hơn 77.6 s và target thấp hơn 0.03. Ô sao lưu Drive thấy thư mục đã tồn tại thì bỏ qua mà không báo lỗi, nên em tải về artefact cũ trộn với verdict mới và phải khôi phục lại từ output notebook. Em đã sửa ô đó để mỗi lần tạo một thư mục mới có timestamp. Từ giờ em kiểm tra nội dung file, không chỉ kiểm tra file có tồn tại.

Nếu có thêm hai giờ, em sẽ lưu đầy đủ output của baseline (b) và fine-tune theo từng mẫu để có đủ ca FT thua baseline, rồi thử replay 1–5% dữ liệu tổng quát với max_length=256, giữ nguyên eval và prompt.

## 8. Nguồn dữ liệu và phạm vi hoàn thành

Log đầy đủ nằm trong colab/Lab21_RUN_ALL.ipynb (lần đánh giá đầy đủ NB2+NB5 mất 1434 giây). Ô sao chép Drive bỏ qua thư mục đã tồn tại, nên baseline, autopsy và qualitative local bị cũ. Em đã khôi phục baseline đầy đủ và bốn dòng autopsy từ output notebook; qualitative chỉ khôi phục được sáu preview đã in. Nguồn gốc ghi ở results/recovery_provenance.json, bản cũ giữ tại submission/evidence_before_recovery/. Không tạo thêm dự đoán và không sửa verdict.

Không nhận B1–B4 (chưa làm NB6, dataset riêng, reasoning mask, rank sweep). Đề nghị xét B5 theo liên kết ở mục 9. Có dùng Claude Code hỗ trợ code và soạn báo cáo theo hướng dẫn VIBE-CODING.md; số liệu được đối chiếu với results/ bằng scripts/verify.py (26 pass, 0 fail).

## 9. Adapter công khai — B5

[Hugging Face Hub](https://huggingface.co/nhatnhoem/lab21-qwen35-4b-triage-lora) · [Commit xuất bản adapter](https://huggingface.co/nhatnhoem/lab21-qwen35-4b-triage-lora/commit/107431329495cb46b1e9dd3d0b49029c79439c20).

Đã xác minh repo truy cập công khai và có adapter_model.safetensors, adapter_config.json, tokenizer, model card cùng kết quả đánh giá. Model card công bố rõ verdict FAILED và giới hạn dữ liệu. Việc chia sẻ adapter không thay đổi phán quyết chất lượng. Theo rubric, đây là bằng chứng để xét điểm thưởng B5 (+2).

## 10. Liên kết nộp phương án B

- Repo cá nhân: https://github.com/MinhNhatUet/Day21-Track3-Finetuning-Lab
- Adapter công khai: https://huggingface.co/nhatnhoem/lab21-qwen35-4b-triage-lora
- Danh mục liên kết: LINKS.md ở thư mục gốc repo.

Repo GitHub chứa báo cáo và artefact kết quả để kiểm tra chéo; trọng số adapter được lưu trên Hugging Face.
