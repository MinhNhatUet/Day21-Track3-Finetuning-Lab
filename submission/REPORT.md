# Lab 21 — LoRA cho ticket CSKH tiếng Việt

**Họ tên:** Ngô Đinh Minh Nhật · **MSSV:** 2A202602569  
**Ngày:** 07/10/2026

> Kết quả chính: đánh giá đủ 50 target và 15 regression. Model FAILED cổng hồi quy vì năng lực tổng quát giảm quá ngưỡng. Một phần artefact được khôi phục từ output notebook do bản sao Drive chưa cập nhật; xem mục 8.

## 1. Lựa chọn và thiết kế

Model unsloth/Qwen3.5-4B, tier T4, GPU Tesla T4 (14.6 GB theo log), precision fp16. Chọn cấu hình mặc định để kiểm tra LoRA trên Colab, giữ cùng base model giữa baseline và adapter. Dataset mặc định có 250 ticket CSKH tiếng Việt, bốn nhãn JSON intent, urgency, product, sentiment. Bài toán chấm được theo nhãn khách quan nên không cần LLM judge.

Log NB1 xác nhận 225 train / 25 validation; mã dùng seed 42. NB3 ghi hai epoch, batch 1, gradient accumulation 16 và 30 optimizer step. Bốn run đều dùng 30 step. Thống kê token ghi p95=98, p99=100, max=101, đề xuất max_length=256. Log xác nhận thực tế dùng 1024 theo tier mặc định. Giới hạn này không cắt mẫu nhưng dư so với phân bố; 256 là lựa chọn cần kiểm tra ở lần tối ưu sau. Không tuyên bố đã đặt max_length theo p95 trong lần chạy hiện tại.

Lần đầu đo tám mẫu trước train. Sau train, chạy lại NB2 và NB5 trên đủ 50 target và 15 regression, giữ nguyên prompt và dữ liệu. Đây là mở rộng đánh giá sau train, không phải baseline đầy đủ đã đóng băng trước train. SHA prompt ở hai lần giống nhau, checksum eval trùng bản gốc; vẫn cần công khai giới hạn thứ tự này để không khẳng định quá mức tính tuân thủ quy trình.


### Giải thích lựa chọn độ dài bằng số đo

Theo thống kê, ngay cả mẫu dài nhất (101 token) cũng ngắn hơn cả 256 lẫn 1024. Trong `build_example`, max_length là ngưỡng cắt, không phải lệnh đệm mọi mẫu lên đúng ngưỡng; mã chỉ cắt khi chuỗi dài hơn giới hạn. Vì vậy, trên 250 mẫu đã đo, riêng bước cắt token cho cùng kết quả ở hai giới hạn này. Batch train bằng 1 và packing tắt theo log; không có cơ sở để nói dùng 1024 khiến số token thực tế tăng bốn lần. Điều này giải thích vì sao cấu hình hiện tại không làm mất câu trả lời, nhưng không biến lựa chọn mặc định thành một thí nghiệm tối ưu bộ nhớ. Khuyến nghị dựa trên p95 vẫn là 256; không nhận đã đo tốc độ hoặc VRAM ở giới hạn đó. Nguồn: token_stats.json, training_evidence.json và src/labkit/data.py.

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

Không nên triển khai bản fine-tune correct theo cổng chất lượng hiện tại. Dù target đạt 0.97, mức suy giảm regression vượt ngưỡng hơn sáu lần và latency cao hơn baseline tối ưu. Với khách hàng thật, lợi ích phân loại ticket không tự động bù nguy cơ trả lời kém ở yêu cầu khác. Một model chỉ phục vụ triage có thể có yêu cầu sản phẩm khác, nhưng thí nghiệm đã đặt cổng bảo toàn năng lực tổng quát nên phải tôn trọng cổng đó, không đổi tiêu chí sau khi biết kết quả.

Điểm quan trọng nhất là mở rộng từ tám mẫu lên tập đầy đủ đã thay đổi phán quyết từ PASSED thành FAILED. Số liệu ban đầu không sai nhưng phạm vi kết luận quá hẹp. Trong đối chứng, giảm LR mười lần làm target và format về 0 ở cùng ngân sách step; vị trí adapter hòa target khi khớp tham số. QLoRA tiết kiệm bộ nhớ nhưng giảm chất lượng và tăng thời gian. Mask đúng là điều kiện nền để tin phép đo, chưa đủ bảo đảm chất lượng sau train. Tôi sẽ ưu tiên phân tích regression từng câu và thử replay tổng quát trên tập phát triển, rồi chấm lại với ngưỡng cũ. Lưu đầy đủ output phải là phần thiết kế thí nghiệm, vì thiếu baseline từng mẫu đã làm mất khả năng phân tích thắng/thua. Kết luận không nên deploy có bằng chứng hữu ích hơn tuyên bố thắng chỉ dựa vào target.

Ba bài học cụ thể:

1. Tám mẫu regression cho kết quả bằng nhau 0.75, nhưng đủ 15 mẫu cho thấy mức giảm khoảng 0.1356; smoke test không đủ kết luận bảo toàn năng lực.
2. Attention-only loss 0.5374 thấp hơn correct 0.6265 nhưng cùng target 0.97; chọn theo training loss không bảo đảm cải thiện tác vụ.
3. Ô sao chép Drive bỏ qua thư mục tồn tại khiến artefact cũ và mới bị trộn; phải kiểm tra nội dung và nguồn, không chỉ file có tồn tại.

Nếu có thêm hai giờ, ưu tiên lưu đầy đủ output, phân nhóm lỗi regression, rồi thử replay 1–5% như giả thuyết của lab với ngân sách ghi rõ. Giữ nguyên eval và prompt, dùng tập phát triển để chọn cấu hình. Chưa có kết quả mới nên không nhận là đã khắc phục hồi quy.

## 8. Nguồn dữ liệu và phạm vi hoàn thành

Notebook colab/Lab21_RUN_ALL.ipynb chứa log lần đầu và lần đánh giá đầy đủ sau đó; lần sau hoàn thành NB2+NB5 trong 1434 giây. Output sao chép Drive báo thư mục đích đã tồn tại và bỏ qua, giải thích vì sao baseline, autopsy và qualitative local còn cũ trong khi verdict đã mới.

Đã khôi phục nguyên JSON baseline đầy đủ và bốn dòng autopsy từ output notebook. Qualitative chỉ khôi phục sáu preview được in. Xem results/recovery_provenance.json; bản cũ lưu tại submission/evidence_before_recovery/. Không tạo thêm dự đoán hoặc sửa verdict FAILED. Baseline tám mẫu được đo trước train; baseline đầy đủ đo sau train, đây là giới hạn quy trình cần công khai.

Chưa có bằng chứng NB6, dataset riêng, đối chứng reasoning mask hoặc rank sweep nên không nhận điểm thưởng B1–B4. Adapter đã công khai trên Hugging Face; đề nghị xét B5 theo liên kết bên dưới. Báo cáo có AI hỗ trợ tổng hợp; phản tư cá nhân cần tác giả xác nhận. Gatekeeper kiểm tra artefact và tính nhất quán cơ bản, không bảo đảm đã đủ mọi yêu cầu rubric, đặc biệt định tính.

## 9. Adapter công khai — B5

[Hugging Face Hub](https://huggingface.co/nhatnhoem/lab21-qwen35-4b-triage-lora) · [Commit xuất bản adapter](https://huggingface.co/nhatnhoem/lab21-qwen35-4b-triage-lora/commit/107431329495cb46b1e9dd3d0b49029c79439c20).

Đã xác minh repo truy cập công khai và có adapter_model.safetensors, adapter_config.json, tokenizer, model card cùng kết quả đánh giá. Model card công bố rõ verdict FAILED và giới hạn dữ liệu. Việc chia sẻ adapter không thay đổi phán quyết chất lượng. Theo rubric, đây là bằng chứng để xét điểm thưởng B5 (+2).

## 10. Liên kết nộp phương án B

- Repo cá nhân: https://github.com/MinhNhatUet/Day21-Track3-Finetuning-Lab
- Adapter công khai: https://huggingface.co/nhatnhoem/lab21-qwen35-4b-triage-lora
- Danh mục liên kết: LINKS.md ở thư mục gốc repo.

Repo GitHub chứa báo cáo và artefact kết quả để kiểm tra chéo; trọng số adapter được lưu trên Hugging Face.
