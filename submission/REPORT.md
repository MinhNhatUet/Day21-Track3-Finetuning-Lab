# Lab 21 — LoRA cho ticket CSKH tiếng Việt

**Họ tên:** Ngô Đinh Minh Nhật · **MSSV:** 2A202602569  
**Ngày:** 07/10/2026

> Kết quả chính: lần chạy cuối đi trọn NB1 → NB5 với đủ 50 target / 15 regression và max_length=256. LoRA thắng baseline (b) 33/50 ticket, hòa 17, thua 0, nhưng FAILED cổng hồi quy vì năng lực tổng quát giảm quá ngưỡng.

## 1. Lựa chọn và thiết kế

Model unsloth/Qwen3.5-4B, tier T4, GPU Tesla T4 (14.6 GB theo log), precision fp16.

**Vì sao chọn model này.** Colab free chỉ có T4 16 GB và T4 không hỗ trợ bf16, nên cần một base đủ nhỏ để LoRA 16-bit chạy fp16 mà vẫn còn chỗ cho đối chứng QLoRA. Qwen3.5-4B vừa khít: peak VRAM của run correct là 8.78 GB, còn dư khoảng 6 GB. Model cũng xử lý tốt tiếng Việt và có chế độ thinking, nên kiểm tra template `<think>` (mục 2) là bài kiểm tra thực tế. Base được giữ nguyên giữa baseline và adapter để mọi chênh lệch đến từ LoRA.

**Vì sao chọn dataset này.** 250 ticket CSKH tiếng Việt, nhãn JSON gồm bốn trường intent, urgency, product, sentiment. Đây là kiểu tác vụ fine-tune có lợi rõ nhất: định dạng đầu ra cố định và tập nhãn đóng mà prompt khó ép hoàn toàn. Nhãn khách quan nên chấm được theo từng trường, không cần LLM judge, và phép so baseline (b) với LoRA không bị nhiễu bởi người chấm.

Log NB1 xác nhận 225 train / 25 validation, seed 42. NB3: hai epoch, batch 1, gradient accumulation 16 (batch hiệu dụng 16 < 32), 30 optimizer step; bốn run đều dùng 30 step.

**max_length.** Thống kê token: p95=98, p99=100, max=101, nên em đặt max_length=256 (p95 làm tròn lên lũy thừa 2, có dư cho template). Tier T4 mặc định là 1024, nên trước lần chạy cuối em sửa dòng `max_length` của tier T4 trong `src/labkit/config.py` bằng `sed` trên Colab. Ô pipeline chạy từng stage trong subprocess nên log NB1 không được lưu vào notebook; đây là khai báo của em. Vì `build_example` chỉ cắt chứ không đệm và mẫu dài nhất chỉ 101 token, 256 hay 1024 đều không cắt mẫu nào, nên kết quả gần như trùng lần chạy đầu ở 1024 (loss correct 0.6257 so với 0.6265, wrong_lr và qlora trùng 4 chữ số).

**Thứ tự đo baseline.** Lần chạy cuối đo baseline trên đủ 50/15 ở NB2 **trước** NB3 (chi tiết ở mục 3).

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

Nguồn: results/baselines_frozen.json, verdict.json. SHA prompt tối ưu: 719e74d3b6232053.

| Run | target | regression | format | latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3242.6 |
| (b) Base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1059.0 |
| (c) LoRA correct | 0.9700 | 0.6556 | 1.0000 | 1414.7 |

Target là độ chính xác trung bình theo trường, không phải tỷ lệ ticket đúng hoàn toàn. Regression là keyword recall trung bình trên 15 câu. Format dùng bộ trích JSON linh hoạt và kiểm tra khóa, không chứng minh output luôn là JSON thuần không kèm văn bản. Prompt tối ưu mạnh hơn naive: target tăng 0.765, format lên 1. Naive có format bằng 0 nên không thể coi target bằng 0 là bằng chứng model không hiểu nội dung.

LoRA tăng 20.5 điểm phần trăm target so với baseline tối ưu, nhưng regression giảm khoảng 13.56 điểm phần trăm. Latency tăng 355.7 ms/mẫu, khoảng 33.6%. Không có bằng chứng prompt bị sửa để làm yếu baseline.


### Bằng chứng baseline có trước huấn luyện

Ô pipeline lần chạy cuối chạy tuần tự NB1 → NB2 → NB3 → NB4 → NB5, mỗi stage một tiến trình. NB2 ghi `results/baseline_predictions.json` (dự đoán (a) và (b) cho đủ 50 ticket) trước khi NB3 bắt đầu train. NB5 kiểm tra lại model, SHA prompt, số mẫu và nội dung từng ticket khớp với file đó trước khi ghép cặp với dự đoán FT.

| Lần chạy | Target / regression | Target (a) | Target (b) | Thời điểm so với train |
|---|---|---:|---:|---|
| Lần đầu (lỡ chạy `eval_limit=8`) | 8 / 8 | 0.0000 | 0.6875 | Trước train |
| Lần cuối | 50 / 15 | 0.0000 | 0.7650 | Trước train |

Cả hai cùng SHA prompt `719e74d3b6232053`, checksum eval khớp bản gốc. Prompt (b) được cố định từ đầu, không chọn lại sau khi thấy adapter.

## 4. Đối chứng cấu hình

Nguồn: results/runs.csv và autopsy.json (lần chạy cuối). Target của cả bốn run chấm trên 50 mẫu.

| Run | Vị trí | r | Tham số trainable | LR | Training loss | Target | Train (s) | Peak VRAM (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 | 32,464,896 | 0.0001 | 0.6257 | 0.97 | 411.1 | 8.78 |
| attn_only | q,v | 283 | 32,456,704 | 0.0001 | 0.5376 | 0.97 | 269.2 | 8.79 |
| wrong_lr | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.00 | 402.0 | 8.78 |
| qlora | text-linear, base 4-bit | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.94 | 474.8 | 3.86 |

**4.1. Vị trí và rank.** Attention-only ít hơn 8,192 tham số, sai lệch khoảng 0.0252%, dưới ngưỡng 5%. Rank 283 là điều chỉnh để khớp ngân sách khi đổi vị trí, không phải quét rank độc lập. Hai run hòa target 0.97 dù attention-only có loss thấp hơn. Thứ tự theo loss phân biệt hai cấu hình nhưng thứ tự theo target không phân biệt, nên không có bằng chứng text-linear thắng q,v ở đây. Attention-only có latency 910.2 ms/mẫu, thấp hơn correct 1414.7; chưa có regression của đối chứng này để kết luận đủ điều kiện thay thế correct. Cần nhiều seed để đánh giá độ ổn định.

**4.2. Learning rate.** Wrong_lr giảm LR mười lần, giữ rank, vị trí, precision và số step. Training loss 1.5702 cao hơn correct 0.6257, target và format bằng 0. Log lần chạy đầu của correct có các mốc loss giảm từ 2.163 xuống 0.02474, nhưng có dao động cuối và một số grad_norm=nan; không khẳng định mọi bước đều ổn định. Cột final_loss được gán từ result.training_loss, không phải riêng loss step cuối. LR thấp phù hợp giả thuyết adapter chưa học đủ trong ngân sách hiện tại; bỏ qua LR có thể khiến ta kết luận nhầm model hoặc LoRA không phù hợp. Lỗi format cũng ảnh hưởng trực tiếp target nên chưa tách được hoàn toàn lỗi ngữ nghĩa.

**4.3. QLoRA.** Peak VRAM giảm 4.92 GB, khoảng 56.0%, nhưng train tăng 63.7 giây, khoảng 15.5%. Target giảm 0.03 và latency tăng từ 1414.7 lên 1795.2 ms/mẫu; format vẫn bằng 1. NB5 nạp base 4-bit cho adapter QLoRA để khớp cấu hình train. Kết quả ủng hộ ưu tiên LoRA 16-bit khi đủ VRAM trên tác vụ này, nhưng chưa chứng minh QLoRA luôn không phù hợp. Tiết kiệm bộ nhớ vẫn hữu ích khi GPU hạn chế; thí nghiệm chỉ có một seed và chưa đo regression của đối chứng.


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

Nguồn: `results/training_evidence.json`, trích nguyên các mốc log Colab của **lần chạy đầu** (cùng cấu hình, seed và số step; log từng step của lần chạy cuối không được lưu vì chạy trong subprocess, nhưng training loss cuối của bốn run giữa hai lần lệch nhau không quá 0.0008). Epoch là giá trị in trong log; loss ở mỗi mốc là giá trị logging của trainer, không phải loss validation.

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

## 6. Định tính: so cặp baseline (b) và LoRA trên từng ticket

Nguồn: `results/qualitative.json` (đủ 50 cặp dự đoán, không cắt ngắn) và `results/baseline_predictions.json`. Điểm là độ chính xác theo trường (0.25 mỗi trường đúng); thắng/hòa/thua so điểm LoRA với điểm (b) trên cùng ticket.

| Kết quả so cặp | Số ticket | Chi tiết điểm (b) → LoRA |
|---|---:|---|
| LoRA thắng | 33 | 0.75 → 1.0: 23 · 0.5 → 1.0: 8 · 0.5 → 0.75: 2 |
| Hòa | 17 | 1.0 = 1.0: 13 · 0.75 = 0.75: 4 |
| LoRA thua | **0** | — |

**Về yêu cầu ≥2 ca fine-tune thua:** trên tập eval đã đóng băng, LoRA **không thua baseline ở ticket nào**. Em không thể đưa ca thua không tồn tại, nên thay vào đó báo cáo đủ phân bố ở trên và đưa các ca LoRA **vẫn sai** hoặc **không hơn** baseline. Chỗ adapter thua thật sự nằm ở regression (mục 5), không nằm ở target.

| i | Ticket | Nhãn đúng (intent / urgency / sentiment) | (b) | LoRA | KQ |
|---|---|---|---|---|---|
| 3 | …bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều. | hoan_tien / thap / tich_cuc | urgency=trung_binh · 0.75 | urgency=trung_binh · 0.75 | Hòa, cả hai sai |
| 39 | …nồi chiên không dầu… Hoàn tiền. **Khi nào tiện.** Quá tệ. | hoan_tien / thap / tieu_cuc | urgency=cao · 0.75 | urgency=trung_binh · 0.75 | Hòa, sai khác kiểu |
| 5 | …nồi chiên không dầu… Thiếu phụ kiện. **Khi nào tiện.** Cho tôi hỏi. | san_pham_loi / thap / trung_tinh | intent=hoan_tien, urgency=cao · 0.5 | urgency=trung_binh · 0.75 | Thắng nhưng vẫn sai |
| 35 | …balo laptop… Hoàn lại. Sớm nhé. Nhờ shop kiểm tra. | doi_tra / trung_binh / trung_tinh | intent=hoan_tien, urgency=cao · 0.5 | đúng hết · 1.0 | Thắng |
| 36 | …tai nghe bluetooth… Giá bao nhiêu. Hỏi cho biết thôi. Rất thất vọng. | hoi_thong_tin / thap / tieu_cuc | intent=doi_tra, urgency=trung_binh · 0.5 | đúng hết · 1.0 | Thắng |
| 49 | …ốp lưng điện thoại… Sai màu. Sớm nhé. Shop xem giúp. | san_pham_loi / trung_binh / trung_tinh | urgency=cao, sentiment=tieu_cuc · 0.5 | đúng hết · 1.0 | Thắng |

Cột (b) và LoRA chỉ ghi các trường sai. Product đúng ở mọi dòng nên không ghi; output LoRA đều là JSON thuần, không kèm văn bản.

**Mẫu lỗi duy nhất của LoRA.** Cả 6 ticket LoRA chưa đạt 1.0 (i = 3, 5, 12, 39, 41, 46) đều sai đúng một trường: urgency `thap` bị đoán thành `trung_binh`. Đó cũng chính là toàn bộ 6 ticket eval có cụm "Khi nào tiện". Điều bất ngờ là tập train có 35 ticket chứa cụm này và cả 35 đều nhãn `thap`, nên đây không phải lỗi thiếu dữ liệu. Giả thuyết của em là sau 30 step adapter học được thiên hướng chung "ticket có vấn đề → trung_binh" mạnh hơn tín hiệu từ cụm từ nhẹ nhàng đứng cuối câu. Muốn kiểm chứng cần đo trên tập dev riêng hoặc train lâu hơn, chưa làm trong lab này. Baseline (b) cũng sai ở cả 6 ticket này, nên đây là chỗ cả prompt lẫn LoRA đều yếu.

Các ca thắng (35, 36, 49) cho thấy LoRA sửa chủ yếu hai lỗi của (b): nhầm intent giữa `hoan_tien` / `doi_tra` / `hoi_thong_tin`, và đẩy urgency lên `cao` khi ticket có "Sớm nhé".

## 7. Kết luận và bài học

**Kết luận: không deploy adapter này.** Target lên 0.97 nhìn rất đẹp, hơn prompt tối ưu 20.5 điểm, nhưng regression tụt từ 0.7911 xuống 0.6556, gấp hơn sáu lần ngưỡng 0.02 mà em đã chấp nhận trước khi chạy. Nếu bây giờ nới ngưỡng thì cả cổng hồi quy mất ý nghĩa. Lý do em tin đây là quên thảm hoạ chứ không phải lỗi pipeline: format vẫn bằng 1, mask đúng (0.41 token được tính loss), và cả 225 mẫu train đều cùng một dạng prompt phân loại JSON. 30 step với LR 1e-4 trên dữ liệu đơn điệu như vậy đủ để model kéo về phía tác vụ hẹp, nên khi hỏi kiến thức chung thì trả lời kém đi. Latency cũng tăng 356 ms/mẫu so với base + prompt (b). Với bài toán triage ticket, prompt (b) đạt 0.765 mà không mất gì; muốn dùng LoRA thì phải trộn thêm 1–5% dữ liệu tổng quát rồi chấm lại với đúng ngưỡng cũ.

**Điều em học được**

1. **Em suýt nộp một kết luận sai.** Lần chạy đầu, em để trống ô `EVAL_LIMIT` và nghĩ là đang chạy full, nhưng log in ra `eval_limit=8`. Với 8 mẫu, regression trước và sau đều 0.75, verdict PASSED, em đã định viết report luôn. Chạy lại đủ 50/15 (thêm 24 phút) thì verdict thành FAILED. Từ giờ việc đầu tiên em làm là đọc dòng config in ra trong log, không tin vào những gì mình nghĩ là đã đặt.

2. **Loss đang giảm không có nghĩa là model đang học được việc.** Run `wrong_lr` có loss giảm từ 2.163 xuống 1.119, nhìn đồ thị thì tưởng ổn, nhưng target và format đều bằng 0. Ngược lại, `attn_only` có loss thấp hơn `correct` (0.5376 so với 0.6257) mà target vẫn chỉ hòa 0.97. Trước lab em hay nhìn loss để chọn run; giờ em chỉ xếp hạng bằng điểm trên tập eval.

3. **Thứ em tưởng là đòn bẩy lại không phải.** Em đoán gắn LoRA vào toàn bộ linear sẽ thắng hẳn q,v. Khi khớp ngân sách tham số (r=283 cho q,v), hai bên hòa nhau. Chỉ đổi LR mười lần mới làm kết quả sụp. Ở quy mô này, LR quan trọng hơn vị trí rất nhiều.

4. **Bài học về hạ tầng.** T4 không có bf16, nên phải chạy fp16 và log có vài `grad_norm=nan`. QLoRA tiết kiệm 4.92 GB VRAM nhưng chạy lâu hơn 63.7 s và target thấp hơn 0.03. Ô sao lưu Drive thấy thư mục đã tồn tại thì bỏ qua mà không báo lỗi, nên lần đầu em tải về artefact cũ trộn với verdict mới và phải khôi phục từ output notebook. Cuối cùng em chạy lại toàn bộ và lưu luôn dự đoán từng mẫu ra file, vì chỉ có số trung bình thì không trả lời được câu hỏi "LoRA thua ở đâu". Từ giờ em kiểm tra nội dung file, không chỉ kiểm tra file có tồn tại.

5. **Dữ liệu train đủ chưa chắc model đã học.** 35 ticket train có "Khi nào tiện" đều nhãn `thap`, vậy mà LoRA sai cả 6 ticket eval có cụm đó. Trước đây gặp lỗi kiểu này em sẽ nghĩ ngay là thiếu dữ liệu; giờ em biết phải đếm trong tập train trước khi kết luận.

Nếu có thêm hai giờ, em sẽ thử replay 1–5% dữ liệu tổng quát để kéo regression về trong ngưỡng, và đo lại urgency trên các ticket "Khi nào tiện" khi train lâu hơn, giữ nguyên eval và prompt.

## 8. Nguồn dữ liệu và phạm vi hoàn thành

Mọi số trong report lấy từ `results/` của **lần chạy cuối** (NB1 → NB5 trên Colab T4, đủ 50/15, max_length=256), trừ bảng diễn biến loss ở mục 4 lấy từ lần chạy đầu. Lần chạy đầu lỡ dùng `eval_limit=8`, rồi bản sao Drive bỏ qua thư mục đã tồn tại nên artefact bị trộn; em đã phải khôi phục từ output notebook. Các file của giai đoạn đó được giữ để đối chiếu: `results/recovery_provenance.json`, `results/training_evidence.json` và `submission/evidence_before_recovery/`. Lần chạy cuối thay thế toàn bộ, không sửa tay số nào và không sửa verdict.

Không nhận B1–B4 (chưa làm NB6, dataset riêng, reasoning mask, rank sweep). Đề nghị xét B5 theo liên kết ở mục 9. Có dùng Claude Code hỗ trợ code và soạn báo cáo theo hướng dẫn VIBE-CODING.md; số liệu được đối chiếu với results/ bằng scripts/verify.py (26 pass, 0 fail).

## 9. Adapter công khai — B5

[Hugging Face Hub](https://huggingface.co/nhatnhoem/lab21-qwen35-4b-triage-lora) · [Commit xuất bản adapter](https://huggingface.co/nhatnhoem/lab21-qwen35-4b-triage-lora/commit/107431329495cb46b1e9dd3d0b49029c79439c20).

Đã xác minh repo truy cập công khai và có adapter_model.safetensors, adapter_config.json, tokenizer, model card cùng kết quả đánh giá. Model card công bố rõ verdict FAILED và giới hạn dữ liệu. Việc chia sẻ adapter không thay đổi phán quyết chất lượng. Theo rubric, đây là bằng chứng để xét điểm thưởng B5 (+2).

## 10. Liên kết nộp phương án B

- Repo cá nhân: https://github.com/MinhNhatUet/Day21-Track3-Finetuning-Lab
- Adapter công khai: https://huggingface.co/nhatnhoem/lab21-qwen35-4b-triage-lora
- Danh mục liên kết: LINKS.md ở thư mục gốc repo.

Repo GitHub chứa báo cáo và artefact kết quả để kiểm tra chéo; trọng số adapter được lưu trên Hugging Face.
