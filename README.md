# Hue Depth Flood Mapping: A Physics-Informed Machine Learning Approach

![GitHub last commit](https://img.shields.io/github/last-commit/ntuson0810-droid/Hue-Depth-Flood-Mapping)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-red.svg)

## 📌 Giới thiệu dự án (Project Overview)
Dự án cá nhân này ứng dụng công nghệ Viễn thám (Remote Sensing), Hệ thống thông tin địa lý (GIS) và các thuật toán Học máy (Machine Learning) để dự báo và thành lập bản đồ độ sâu ngập lụt tại khu vực tỉnh Thừa Thiên Huế. Lấy sự kiện đại hồng thủy lịch sử tháng 11/1999 làm kịch bản tham chiếu chính (với 1.000 điểm khảo sát thực địa), dự án chuyển đổi từ bài toán phân loại ngập/không ngập truyền thống sang **bài toán hồi quy định lượng** để ước lượng trực tiếp giá trị độ sâu ngập lụt tại từng điểm ảnh 30m.

Điểm nổi bật của dự án là áp dụng khung tiếp cận **Physics-Informed Machine Learning (Học máy tích hợp tri thức vật lý)**:
1. Tích hợp hệ số nhám Manning's N từ dữ liệu lớp phủ bề mặt ESA WorldCover.
2. Áp dụng 11 ràng buộc đơn điệu (monotonic constraints) trong XGBoost để đảm bảo tính nhất quán vật lý (ví dụ: độ cao địa hình tăng $\rightarrow$ độ sâu ngập giảm).
3. Hậu xử lý thủy lực (hydraulic post-processing) theo 4 vùng địa hình đặc thù dựa trên nguyên lý *cap-only* và *bathtub spreading*.

---

## 🔄 Workflow (Luồng Quy trình)

Quy trình phát triển dự án được thiết kế chặt chẽ từ khâu chuẩn bị dữ liệu đến xuất bản đồ cuối cùng:

1. **Thu thập & Tiền xử lý (GIS & Remote Sensing)**:
   - Đồng bộ hóa độ phân giải không gian (30m) và hệ tọa độ (UTM Zone 48N) cho toàn bộ ảnh Raster từ các nguồn ALOS, ESA, JRC.
   - Trích xuất giá trị Raster tại 1.000 tọa độ vết lũ (FloodMarks).
2. **Feature Engineering**:
   - Tính toán 16 đặc trưng gốc (Slope, TWI, HAND...).
   - Xây dựng 15 đặc trưng tương tác thủy lực (ví dụ: `manning_n` $\times$ `hand30_100`).
3. **Mô hình hóa (Machine Learning)**:
   - Phân chia dữ liệu theo tỷ lệ 80% Train, 10% Validation, 10% Test (có phân tầng theo cấp độ ngập).
   - Huấn luyện 3 kiến trúc: Random Forest, XGBoost-GPU, Deep Neural Network.
4. **Đánh giá & Hậu xử lý**:
   - Đánh giá trên các chỉ số: $R^2$, RMSE, MAE, MAPE, Accuracy phân loại 6 cấp.
   - Hậu xử lý không gian (Gaussian smoothing, giới hạn độ sâu tối đa theo địa hình).
5. **Thành lập bản đồ**: Xuất bản đồ GeoTIFF liên tục và bản đồ phân vùng rủi ro 6 cấp.

---

## 🗂 Bộ dữ liệu chi tiết (Datasets)

Dự án sử dụng 31 đặc trưng đầu vào, được khai thác và xử lý từ 5 nguồn dữ liệu viễn thám và thực địa uy tín:

| Dữ liệu gốc | Nguồn cung cấp | Độ phân giải | Các đặc trưng trích xuất / dẫn xuất |
|---|---|---|---|
| **ALOS World 3D** | JAXA | 30m | DSM gốc, Độ dốc (Slope), Hướng sườn (Aspect), Độ cong (Curvature), Độ gồ ghề (Roughness), TPI, Chỉ số ẩm TWI. |
| **MERIT-Hydro** | Yamazaki (2019) | 30m | **HAND** (Height Above Nearest Drainage) - Độ cao tương đối so với dòng chảy, Tích lũy dòng chảy (Flow Acc), Sức mạnh dòng chảy (SPI). |
| **ESA WorldCover 2021** | ESA | 10m $\rightarrow$ 30m | Lớp phủ bề mặt (LULC), chuyển đổi thành **Hệ số nhám Manning's N** qua bảng tra cứu thủy lực. |
| **JRC Global Surface Water** | JRC / Ủy ban Châu Âu | 30m | Xác suất xuất hiện bề mặt nước lịch sử (`gsw_occure`). |
| **Lượng mưa trạm** | Đài KTTV Trung Trung Bộ | 5 trạm | Tổng lượng mưa tích lũy 11/1999 (nội suy không gian bằng phương pháp IDW). |
| **FloodMarks 1999** | Khảo sát thực địa | Điểm (Point) | **Biến mục tiêu (Target)**: Độ sâu ngập lụt tại 1.000 vị trí (Zero-inflated: 715 điểm 0m, 285 điểm > 0m). |

---

## 🛠 Các mô hình được phát triển

1. **Random Forest (RF - v5)**: Thuật toán baseline Ensemble mạnh mẽ, đạt kết quả tốt nhất về độ chính xác tổng thể nhờ khả năng chống nhiễu trên dữ liệu zero-inflated.
2. **XGBoost (Extreme Gradient Boosting - GPU v5)**: Tích hợp 11 ràng buộc đơn điệu (monotonic constraints) đảm bảo quy luật thủy văn, kết hợp huấn luyện siêu tốc bằng GPU.
3. **Deep Neural Network (DNN - v6)**: Kiến trúc đặc biệt sử dụng hàm kích hoạt Sigmoid ở lớp đầu ra tạo giới hạn `Bounded Output [0, 5m]`, tối ưu cực tốt việc phân biệt điểm không ngập và có ngập.

---

## 📊 Hình ảnh Kết quả Thực nghiệm

Dưới đây là so sánh trực quan kết quả giữa dự báo của mô hình (Predicted) và Thực tế (Actual), cũng như Bản đồ phân vùng rủi ro ngập lụt 6 cấp cho toàn bộ tỉnh Thừa Thiên Huế.

### 1. Mô hình Random Forest ($R^2 = 0.8946$)
Bản đồ dự báo mượt mà và phân hóa rất tốt ở các vùng ngập từ 0.5m đến 2m.

<div align="center">
  <img src="results/KQ%20RF/RFv5_Predicted_vs_Actual.png" width="45%" />
  <img src="results/KQ%20RF/RF_PHAN_LOAI.png" width="45%" />
</div>

### 2. Mô hình XGBoost - Physics-Informed ($R^2 = 0.8884$)
Dự báo mang tính "quyết liệt" hơn do áp dụng chặt chẽ các quy luật bảo toàn vật lý.

<div align="center">
  <img src="results/KQ%20XGB/XGBv51_Predicted_vs_Actual.png" width="45%" />
  <img src="results/KQ%20XGB/XGB_PHAN_LOAI.png" width="45%" />
</div>

### 3. Mô hình Deep Neural Network ($R^2 = 0.8847$)
Xử lý xuất sắc các điểm Không Ngập nhờ cấu trúc Sigmoid giới hạn độ sâu đầu ra.

<div align="center">
  <img src="results/KQ%20DNN/DNNv6_Predicted_vs_Actual.png" width="45%" />
  <img src="results/KQ%20DNN/DNN_PHAN_LOAI.png" width="45%" />
</div>

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
Do giới hạn dung lượng trên Github, các file Raster gốc kích thước lớn (`.tif`) không được tải lên. Đặt các file Raster tương ứng vào thư mục `data/raster/` theo cấu hình `src/core/utils.py`.

### 3. Huấn luyện và dự đoán
Mã nguồn đã được mô-đun hóa (Clean Code). Mở Terminal tại thư mục gốc và chạy:

```bash
python src/rfmanningsv5.py
python src/xgbmanningsv5.py
python src/dnnmanningv5.py
```
