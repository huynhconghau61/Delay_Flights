# ỨNG DỤNG MÔ HÌNH HỌC MÁY TRONG DỰ ĐOÁN TÌNH TRẠNG TRỄ GIỜ BAY
+ Dataset: https://www.kaggle.com/datasets/usdot/flight-delays?select=airports.csv
+ Mục tiêu đề tài:
  - Phát triển một mô hình học máy có khả năng dự đoán tình trạng trễ giờ bay một cách chính xác
  - Sử dụng kỹ thuật tiền xử lý dữ liệu, chọn lựa và tinh chỉnh các thuật toán để cải thiện độ chính xác và hiệu quả của mô hình.
  - Thử nghiệm và đánh giá: Kiểm tra mô hình trên các tập dữ liệu thử nghiệm để đánh giá độ chính xác, độ phủ, F1 score, và ROC-AUC. So sánh hiệu suất giữa các mô hình học máy khác nhau để xác định phương án tối ưu nhất.
  - Áp dụng mô hình vào thực tiễn để hỗ trợ các hãng hàng không trong việc quản lý lịch trình bay và giảm thiểu tình trạng trễ giờ bay…
+ Phương pháp thực hiện
  - Thu thập dữ liệu: Sử dụng tập tin flights.csv cho dữ liệu về chuyến bay và file airports.csv cho thông tin về sân bay, bao gồm thời gian, số hiệu chuyến bay, mã sân bay xuất phát và đến, thời gian dự kiến và thực tế cất cánh/hạ cánh, độ trễ và các thông tin liên quan khác.
  - Tiền xử lý dữ liệu: Làm sạch dữ liệu bằng cách loại bỏ các giá trị thiếu, loại bỏ nhiễu, và chuyển đổi dữ liệu thành định dạng phù hợp cho việc phân tích. Điều này bao gồm việc mã hóa dữ liệu phân loại và chuẩn hóa các biến số liên tục.
  - Phân tích tương quan: Xác định mức độ ảnh hưởng của các yếu tố đến độ trễ chuyến bay, sử dụng phương pháp phân tích tương quan và kỹ thuật thống kê khác.
  -  Lựa chọn và kỹ thuật giảm chiều dữ liệu: Sử dụng kỹ thuật như phân tích thành phần chính (PCA) để giảm số lượng tính năng, tránh hiện tượng quá mức (overfitting) và cải thiện hiệu suất của mô hình.
  -  Lựa chọn mô hình: Thử nghiệm với các mô hình học máy khác nhau như cây quyết định, rừng ngẫu nhiên, mạng neural, SVM, và mô hình tăng cường gradient.
  -  Đào tạo mô hình: Sử dụng dữ liệu đã được tiền xử lý để đào tạo các mô hình, áp dụng kỹ thuật chia tập dữ liệu thành tập huấn luyện và tập kiểm tra để đánh giá mô hình một cách khách quan.
  -  Đánh giá mô hình: Sử dụng các chỉ số đánh giá như độ chính xác, F1 score, ROC-AUC để đánh giá hiệu suất của mô hình trên tập dữ liệu kiểm tra.
  -  Triển khai thực tế: Tích hợp mô hình vào hệ thống IT của hãng hàng không hoặc sân bay để cung cấp dự đoán độ trễ chuyến bay trong thời gian thực.
+ Bảng đánh giá mô hình sau khi đã được huấn luyện
![image](https://github.com/user-attachments/assets/69397db7-2eab-4199-9dd4-24c4143647ef)


