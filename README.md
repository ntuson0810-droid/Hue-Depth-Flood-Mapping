# Hue Depth Flood Mapping: A Physics-Informed Machine Learning Approach

![GitHub last commit](https://img.shields.io/github/last-commit/ntuson0810-droid/Hue-Depth-Flood-Mapping)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-red.svg)

> **Thông tin Nhóm nghiên cứu:**
> - **Tác giả:** Nguyễn Tư Sơn
> - **Đồng tác giả:** TS. Hà Minh Cường & ThS. Hoàng Tích Phúc
> - **Đơn vị công tác:** Khoa Công nghệ Hàng không Vũ trụ, Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội (UET - VNU)

## 📌 Giới thiệu dự án (Project Overview)
Dự án ứng dụng công nghệ Viễn thám (Remote Sensing), Hệ thống thông tin địa lý (GIS) và các thuật toán Học máy (Machine Learning) để dự báo và thành lập bản đồ độ sâu ngập lụt tại khu vực tỉnh Thừa Thiên Huế. Lấy sự kiện đại hồng thủy lịch sử tháng 11/1999 làm kịch bản tham chiếu chính (với 1.000 điểm khảo sát thực địa), dự án chuyển đổi từ bài toán phân loại ngập/không ngập truyền thống sang **bài toán hồi quy định lượng** để ước lượng trực tiếp giá trị độ sâu ngập lụt tại từng điểm ảnh 30m.

Điểm nổi bật của dự án là áp dụng khung tiếp cận **Physics-Informed Machine Learning (Học máy tích hợp tri thức vật lý)**:
1. Tích hợp hệ số nhám Manning's N từ dữ liệu lớp phủ bề mặt ESA WorldCover.
2. Áp dụng 11 ràng buộc đơn điệu (monotonic constraints) trong XGBoost để đảm bảo tính nhất quán vật lý (ví dụ: độ cao địa hình tăng $\rightarrow$ độ sâu ngập giảm).
3. Hậu xử lý thủy lực (hydraulic post-processing) theo 4 vùng địa hình đặc thù dựa trên nguyên lý *cap-only* và *bathtub spreading*.

---

## 🔄 Sơ đồ Quy trình Nghiên cứu (Workflow)

Toàn bộ luồng dữ liệu từ khâu thu thập viễn thám đến khi xuất bản đồ được hệ thống hóa qua biểu đồ dưới đây:

```mermaid
flowchart TD
    subgraph GIS ["GIS - REMOTE SENSING"]
        direction LR
        subgraph FI ["Flood Inventory"]
            F_Marks["FloodMarks 1999"]
        end
        
        subgraph DR ["Đặc trưng raster đầu vào"]
            direction LR
            D_DEM["DEM (ALOS\nWorld 3D,\n30m)"]
            D_Topo["Đặc trưng địa\nhình (7)"]
            D_Hydro["Đặc trưng\nthủy văn (5)"]
            D_Man["Hệ số\nManning n"]
            D_GSW["GSW\nOccurrence"]
            D_Precip["Precipitation\n1999"]
        end
    end

    subgraph ML ["MACHINE LEARNING"]
        direction TB
        FE["Feature Engineering\n(16 gốc + 15 tương tác =\n31 đặc trưng)"]
        Split["Phân chia dữ liệu (Stratified)\nTrain 80% - Val 10% - Test 10%"]
        
        subgraph Train ["Quá trình Huấn luyện & Tối ưu"]
            direction TB
            Models["1. Random Forest\n2. XGBoost GPU\n3. Deep Neural Network"]
            CV{"Cross-Validation (K=5)\nTinh chỉnh tham số"}
            
            Models -->|Input parameters| CV
            CV -->|R² < 0.85 (No)| Models
            
            Trained{{"Mô hình đã huấn luyện:\nRF, XGBoost, DNN"}}
            CV ---> Trained
        end
        
        Eval["Đánh giá hiệu suất\n(R², RMSE, MAE, MAPE, Accuracy)"]
        Post["Hậu xử lý thủy lực\n(Gaussian smoothing & Phân vùng)"]
        
        FE --> Split
        Split --> Train
        Trained --> Eval
        Eval --> Post
    end

    Map[/"BẢN ĐỒ ĐỘ SÂU NGẬP LỤT\nĐộ phân giải 30m (~5,8 triệu pixel)\nPhân vùng 6 cấp độ ngập"/]

    F_Marks --> Split
    DR --> FE
    Post --> Map

    style GIS fill:#f9faff,stroke:#3b5998,stroke-width:2px,color:#1d3557
    style ML fill:#f4fff8,stroke:#2a9d8f,stroke-width:2px,color:#1d3557
    style Map fill:#f4a261,stroke:#e76f51,stroke-width:2px,color:#fff
    style FI fill:#fff,stroke:#3b5998
    style DR fill:#fff,stroke:#3b5998
    style Train fill:#e8f8f5,stroke:#2a9d8f,stroke-dasharray: 5 5
    style CV fill:#f4a261,stroke:#e76f51,color:#fff
    style Trained fill:#cbaacb,stroke:#835090
    style F_Marks fill:#e1f5fe
    style D_DEM fill:#e1f5fe
    style D_Topo fill:#e1f5fe
    style D_Hydro fill:#e1f5fe
    style D_Man fill:#e1f5fe
    style D_GSW fill:#e1f5fe
    style D_Precip fill:#e1f5fe
    style FE fill:#e1f5fe
    style Split fill:#e1f5fe
    style Models fill:#e1f5fe
    style Eval fill:#e1f5fe
    style Post fill:#e1f5fe
```

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
