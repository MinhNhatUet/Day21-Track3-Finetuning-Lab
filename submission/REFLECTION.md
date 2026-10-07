# Reflection — Lab 21

**Ngô Đinh Minh Nhật — 2A202602569**

> Bản nháp dựa trên bằng chứng, có AI hỗ trợ diễn đạt. Tác giả xác nhận phần trải nghiệm cá nhân trước khi nộp.

**1. Điều đáng chú ý nhất**

Đánh giá tám mẫu PASSED nhưng đánh giá đầy đủ FAILED: target đạt 0.97 trong khi regression giảm từ 0.7911 xuống 0.6556. Chạy nhanh đủ kiểm tra pipeline nhưng không đủ bảo đảm model giữ năng lực tổng quát.

**2. Phần tốn thời gian**

QLoRA train lâu nhất trong bốn run: 475.0 giây. Lần đánh giá đầy đủ NB2+NB5 mất 1434 giây theo log. Không có số đo thời gian cá nhân dành cho gỡ lỗi hoặc chờ Colab; tác giả cần xác nhận phần nào tốn công nhất so với dự đoán.

**3. Bài học về fine-tuning**

Target tăng không đủ kết luận nên deploy. Attention-only loss thấp hơn correct nhưng target hòa 0.97. Cần đo mục tiêu sản phẩm, hồi quy và chi phí thay vì chỉ nhìn training loss.

**4. Vai trò AI assistant**

AI hỗ trợ đọc yêu cầu, kiểm tra artefact và soạn báo cáo. Báo cáo đầu dùng tám mẫu vì đó là dữ liệu hiện có; nhận notebook mới phải cập nhật kết luận sang FAILED. AI tìm được lỗi sao chép Drive bỏ qua thư mục tồn tại và khôi phục số liệu từ output. Không có đủ bằng chứng để bịa output baseline hoặc gọi ca sai nhãn là ca thua baseline. Trải nghiệm dùng AI trong giai đoạn train cần tác giả bổ sung nếu có.

**5. Bước đầu cho khách hàng thật**

Đặt trước metric, ngưỡng hồi quy và chi phí lỗi; đo baseline có prompt tốt trước train trên tập đánh giá tách biệt. Lưu output đầy đủ và phiên bản artefact để giải thích lỗi. Thử replay hoặc điều chỉnh trên tập phát triển rồi dùng cổng cố định để quyết định triển khai.
