# Nghiên cứu ứng dụng Viễn thám và Học máy thành lập bản đồ độ sâu ngập lụt tại Thừa Thiên Huế

![GitHub last commit](https://img.shields.io/github/last-commit/ntuson0810-droid/Hue-Depth-Flood-Mapping)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-red.svg)

## 📌 Giới thiệu dự án
Dự án này ứng dụng công nghệ Viễn thám (Remote Sensing) kết hợp với các thuật toán Học máy (Machine Learning) và Học sâu (Deep Learning) để mô phỏng, dự đoán và thành lập bản đồ độ sâu ngập lụt tại khu vực tỉnh Thừa Thiên Huế. Dựa trên dữ liệu của các trận lũ lịch sử (điển hình là trận lũ năm 1999) cùng với hàng loạt các biến số không gian, dự án phát triển các mô hình dự báo không gian nhằm hỗ trợ ra quyết định trong công tác phòng chống thiên tai.

## 🛠 Các mô hình được phát triển
Dự án triển khai và so sánh 3 kiến trúc mô hình chính:
1. **Deep Neural Network (DNN - v6)**: Sử dụng TensorFlow/Keras, được tinh chỉnh hàm kích hoạt (Sigmoid scaled), chặn khoảng giá trị ngập (Bounded Output 0-5m, clip tại 3.5m để phù hợp thực tế) và tăng cường Dropout (0.3) chống ngoại suy sai.
2. **XGBoost (Extreme Gradient Boosting - v5)**: Mô hình dạng cây dựa trên Gradient Boosting, xử lý mạnh mẽ dữ liệu dạng bảng có tương tác phi tuyến phức tạp.
3. **Random Forest (RF - v5)**: Thuật toán Ensemble learning ổn định, dùng để so sánh làm đường cơ sở (baseline) cũng như đánh giá tầm quan trọng của các biến (Feature Importance).

## 🗂 Dữ liệu đầu vào (Features)
Mô hình sử dụng một bộ kết hợp các lớp dữ liệu không gian định dạng Raster (30m/10m resolution):
* **Địa hình (Topographic)**: DEM/DSM, Slope, Aspect, Curvature, Hillshade, Roughness, SPI, TPI, TWI.
* **Thủy văn (Hydrological)**: Flow accumulation, HAND (Height Above Nearest Drainage), Distance to River, GSW (Global Surface Water), Density.
* **Khí tượng (Meteorological)**: Tổng lượng mưa (Precipitation total 1999).
* **Mặt phủ (Land Cover)**: VLUCD (Land use / LULC), Manning's N (độ nhám bề mặt).
* **Ground Truth**: Các mốc ngập lụt khảo sát thực tế (FloodMarks1999.csv).

## 📂 Cấu trúc thư mục

```text
Hue-Depth-Flood-Mapping/
├── data/
│   ├── raw/           # Dữ liệu dạng bảng (CSV, Excel mốc ngập)
│   ├── spatial/       # Dữ liệu không gian vector (Shapefile đường, sông ngòi, ranh giới)
│   └── raster/        # Chứa dữ liệu DEM, LULC, Mưa, Các chỉ số địa hình (Bỏ qua trên Github do kích thước lớn)
├── src/               # Mã nguồn Python cho từng thuật toán
│   ├── dnnmanningv5.py / dnnnomannings.py
│   ├── rfmanningsv5.py / rfnomannings.py
│   └── xgbmanningsv5.py / xgbnomannings.py
├── results/           # Kết quả đầu ra của các mô hình
│   ├── KQ DNN/        # Bản đồ tif, đồ thị huấn luyện, loss, residuals, feature importance
│   ├── KQ RF/         # Các báo cáo đánh giá (RMSE, MAE, R2)
│   └── KQ XGB/
├── docs/              # Tài liệu tham khảo dự án
├── .gitignore
└── README.md
```

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Chuẩn bị môi trường
Khuyến nghị sử dụng **Miniconda** hoặc **Anaconda** để quản lý môi trường.

```bash
# Tạo môi trường ảo
conda create -n flood python=3.9
conda activate flood

# Cài đặt các thư viện cần thiết
pip install pandas numpy rasterio geopandas scikit-learn xgboost tensorflow matplotlib tqdm
```

### 2. Chuẩn bị dữ liệu
Do giới hạn dung lượng trên Github, các file Raster kích thước lớn (`.tif`) không được tải lên kho lưu trữ này. 
* Bạn cần chuẩn bị các file Raster tương ứng và đặt vào thư mục `data/raster/`.
* Kiểm tra và đảm bảo các đường dẫn (relative paths) trong mã nguồn Python ở thư mục `src/` khớp với tên file hiện có.

### 3. Huấn luyện và dự đoán
Mở Terminal/PowerShell tại thư mục gốc của dự án và chạy các script mô hình tương ứng:

```bash
# Chạy mô hình XGBoost
python src/xgbmanningsv5.py

# Chạy mô hình Deep Neural Network
python src/dnnmanningv5.py
```

Mô hình sẽ tự động trích xuất các giá trị pixel, huấn luyện, tạo ra các báo cáo đánh giá (`Model_Evaluation_Report.txt`), biểu đồ `Predicted_vs_Actual.png`, và xuất bản đồ kết quả độ sâu cuối cùng dưới định dạng `.tif` vào thư mục `results/`.

## 📊 Kết quả đầu ra
* **Bản đồ thô (RAW `.tif`)**: Chứa giá trị độ sâu ngập liên tục tính bằng mét (m).
* **Bản đồ phân loại (Classified `.tif`)**: Độ sâu ngập được phân cấp theo các ngưỡng rủi ro.
* **Chỉ số đánh giá**: RMSE (Root Mean Square Error), MAE (Mean Absolute Error), và hệ số xác định $R^2$.
* **Phân tích biến (Feature Importance)**: Tầm quan trọng của biến số dựa trên Permutation và Gain.
