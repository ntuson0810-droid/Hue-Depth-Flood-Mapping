# Nghiên cứu ứng dụng Viễn thám và Học máy trong thành lập bản đồ độ sâu ngập lụt tại tỉnh Thừa Thiên Huế

![GitHub last commit](https://img.shields.io/github/last-commit/ntuson0810-droid/Hue-Depth-Flood-Mapping)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-red.svg)

> **Thông tin Đồ án:**
> - **Tác giả:** Nguyễn Tư Sơn
> - **Giảng viên hướng dẫn:** TS. Hà Minh Cường & ThS. Hoàng Tích Phúc
> - **Đơn vị:** Viện Hàng không Vũ trụ, Trường Đại học Công Nghệ - Đại học Quốc gia Hà Nội (UET - VNU)
> - **Khóa luận tốt nghiệp Đại học (2026)**

## 📌 Giới thiệu dự án
Dự án này ứng dụng công nghệ Viễn thám (Remote Sensing) kết hợp với Hệ thống thông tin địa lý (GIS) và các thuật toán Học máy (Machine Learning) để mô phỏng, dự báo và thành lập bản đồ độ sâu ngập lụt tại khu vực tỉnh Thừa Thiên Huế. Lấy sự kiện đại hồng thủy lịch sử tháng 11/1999 làm kịch bản tham chiếu chính (với 1.000 điểm khảo sát thực địa), dự án chuyển đổi từ bài toán phân loại ngập/không ngập truyền thống sang **bài toán hồi quy định lượng** để ước lượng trực tiếp giá trị độ sâu ngập lụt tại từng điểm ảnh 30m.

Điểm đột phá của dự án là việc áp dụng khung phương pháp **Physics-informed Machine Learning (Học máy tích hợp tri thức vật lý)** thông qua:
1. Tích hợp hệ số nhám Manning's N từ dữ liệu lớp phủ bề mặt ESA WorldCover.
2. Thiết kế 15 đặc trưng tương tác thủy lực (ví dụ: sức cản, độ dẫn nước).
3. Áp dụng 11 ràng buộc đơn điệu (monotonic constraints) trong XGBoost để đảm bảo tính nhất quán vật lý (ví dụ: địa hình càng cao thì độ sâu ngập càng giảm).
4. Xây dựng quy trình hậu xử lý thủy lực (hydraulic post-processing) theo 4 vùng địa hình đặc thù dựa trên nguyên lý *cap-only* và *bathtub spreading*.

## 🛠 Các mô hình được phát triển
Dự án triển khai và so sánh 3 kiến trúc mô hình chính:
1. **Random Forest (RF - v5)**: Thuật toán baseline Ensemble mạnh mẽ, đạt kết quả tốt nhất về độ chính xác tổng thể nhờ khả năng chống nhiễu trên dữ liệu zero-inflated.
2. **XGBoost (Extreme Gradient Boosting - GPU v5)**: Đóng vai trò trung tâm nhờ tích hợp 11 ràng buộc đơn điệu (monotonic constraints) đảm bảo quy luật thủy văn, kết hợp huấn luyện siêu tốc bằng GPU.
3. **Deep Neural Network (DNN - v6)**: Sử dụng TensorFlow/Keras với 4 lớp ẩn. Kiến trúc đặc biệt sử dụng hàm kích hoạt Sigmoid ở lớp đầu ra tạo giới hạn `Bounded Output [0, 5m]`, tối ưu cực tốt việc phân biệt điểm không ngập và có ngập.

*Kết quả nghiên cứu độc lập kiểm chứng nghiên cứu của Grinsztajn et al. (NeurIPS 2022) rằng các mô hình tree-based (RF, XGBoost) vẫn duy trì ưu thế so với Deep Learning trên dữ liệu dạng bảng (tabular data) có kích thước vừa và nhỏ.*

## 🗂 Dữ liệu đầu vào (31 Đặc trưng)
Mô hình sử dụng một bộ kết hợp các lớp dữ liệu không gian định dạng Raster (30m resolution):
* **Địa hình (Topographic)**: DEM ALOS World 3D, Slope, Aspect, Curvature, Hillshade, Roughness, SPI, TPI, TWI.
* **Thủy văn (Hydrological)**: Flow accumulation, HAND (Height Above Nearest Drainage), Distance to River, GSW (Global Surface Water), Density.
* **Khí tượng (Meteorological)**: Tổng lượng mưa nội suy từ 5 trạm đo (IDW).
* **Mặt phủ (Land Cover)**: ESA WorldCover (10m $\rightarrow$ 30m), Hệ số nhám Manning's N.
* **Ground Truth**: 1.000 điểm vết lũ thực địa (80% Train, 10% Val, 10% Test).

---

## 📊 Hình ảnh Kết quả Thực nghiệm

Dưới đây là so sánh trực quan kết quả giữa dự báo của mô hình (Predicted) và Thực tế (Actual), cũng như Bản đồ phân vùng rủi ro ngập lụt 6 cấp cho toàn bộ tỉnh Thừa Thiên Huế.

### 1. Mô hình Random Forest
Đạt hiệu năng tổng thể tốt nhất ($R^2 = 0.8946$, RMSE $= 0.267m$). Bản đồ dự báo mượt mà và phân hóa rất tốt ở các vùng ngập từ 0.5m đến 2m.

<div align="center">
  <img src="results/KQ%20RF/RFv5_Predicted_vs_Actual.png" width="45%" />
  <img src="results/KQ%20RF/RF_PHAN_LOAI.png" width="45%" />
</div>

### 2. Mô hình XGBoost (Physics-Informed)
Tập trung mạnh mẽ vào các đặc trưng DEM do bị áp đặt ràng buộc vật lý ($R^2 = 0.8884$, RMSE $= 0.275m$). Dự báo mang tính "quyết liệt" hơn, vạch rõ ranh giới các vùng ngập cực sâu (>2m).

<div align="center">
  <img src="results/KQ%20XGB/XGBv51_Predicted_vs_Actual.png" width="45%" />
  <img src="results/KQ%20XGB/XGB_PHAN_LOAI.png" width="45%" />
</div>

### 3. Mô hình Deep Neural Network
Xử lý xuất sắc các điểm Không Ngập nhờ cấu trúc Sigmoid giới hạn ($R^2 = 0.8847$). Bản đồ dự báo an toàn hơn với hiện tượng over-flooding được kiểm soát thông qua quy trình hậu xử lý.

<div align="center">
  <img src="results/KQ%20DNN/DNNv6_Predicted_vs_Actual.png" width="45%" />
  <img src="results/KQ%20DNN/DNN_PHAN_LOAI.png" width="45%" />
</div>

> **Kết luận Feature Importance:** Phân tích từ mô hình cho thấy Độ cao địa hình (DEM) và các biến tương tác liên quan (`dem_x_twi`, `dem_squared`, `dem_x_slope`) chiếm gần 80% tầm quan trọng, khẳng định đây là yếu tố chi phối tuyệt đối nguy cơ ngập lụt.

---

## 📂 Cấu trúc thư mục

```text
Hue-Depth-Flood-Mapping/
├── data/
│   ├── raw/           # Dữ liệu bảng (CSV, Excel mốc ngập)
│   ├── spatial/       # Dữ liệu không gian vector
│   └── raster/        # Chứa dữ liệu DEM, LULC, Mưa (Bỏ qua trên Github do kích thước >100MB)
├── src/
│   ├── core/          # MODULE TÁI CẤU TRÚC (Chứa toàn bộ logic không gian, thủy văn dùng chung)
│   ├── dnnmanningv5.py / dnnnomannings.py
│   ├── rfmanningsv5.py / rfnomannings.py
│   └── xgbmanningsv5.py / xgbnomannings.py
├── results/           # Kết quả đầu ra (bản đồ .tif, đồ thị đánh giá, weights)
└── README.md
```

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Chuẩn bị môi trường
Khuyến nghị sử dụng **Miniconda** để quản lý môi trường.

```bash
conda create -n flood python=3.9
conda activate flood
pip install pandas numpy rasterio geopandas scikit-learn xgboost tensorflow matplotlib tqdm astunparse
```

### 2. Chuẩn bị dữ liệu
Do giới hạn dung lượng trên Github, các file Raster gốc kích thước lớn (`.tif`) không được tải lên kho lưu trữ này. 
* Đặt các file Raster tương ứng vào thư mục `data/raster/`.

### 3. Huấn luyện và dự đoán
Mã nguồn đã được mô-đun hóa cực kỳ sạch sẽ (Clean Code). Mở Terminal/PowerShell tại thư mục gốc và chạy:

```bash
# Chạy mô hình Random Forest
python src/rfmanningsv5.py

# Chạy mô hình XGBoost
python src/xgbmanningsv5.py

# Chạy mô hình Deep Neural Network
python src/dnnmanningv5.py
```
