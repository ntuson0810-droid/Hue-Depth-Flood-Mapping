# Nghiên cứu ứng dụng Viễn thám và Học máy trong việc thành lập bản đồ độ sâu ngập lụt tại Huế

Dự án này ứng dụng các mô hình học máy (Machine Learning) và dữ liệu viễn thám để dự đoán và thành lập bản đồ độ sâu ngập lụt tại tỉnh Thừa Thiên Huế.

## Các mô hình sử dụng
- **Random Forest (RF)**
- **XGBoost (XGB)**
- **Deep Neural Network (DNN)**

## Cấu trúc thư mục
- data/: Chứa dữ liệu đầu vào.
  - aw/: Dữ liệu bảng (CSV, Excel) chứa các điểm ngập lụt (ví dụ: trận lũ lịch sử 1999).
  - spatial/: Dữ liệu không gian vector (Shapefile đường, sông ngòi, ranh giới...).
  - aster/: Dữ liệu viễn thám, DEM, ảnh vệ tinh (đã được cấu hình bỏ qua trên GitHub do dung lượng lớn).
- src/: Mã nguồn Python huấn luyện và đánh giá mô hình.
- esults/: Kết quả đầu ra (bản đồ ngập lụt .tif, biểu đồ đánh giá mô hình). Dữ liệu dự đoán dạng bảng lớn đã được cấu hình .gitignore.
- docs/: Tài liệu tham khảo của dự án.

## Hướng dẫn sử dụng
1. Cài đặt các thư viện cần thiết (có thể dùng Miniconda).
2. Chuẩn bị dữ liệu raster đầu vào đặt vào thư mục data/raster/.
3. Chạy các kịch bản trong thư mục src/ (ví dụ: python src/dnnmanningv5.py).

## Tác giả
[Tên của bạn]
