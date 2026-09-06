import pandas as pd
import numpy as np
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==============================================================================
# 👇 KHU VỰC CẤU HÌNH ĐƯỜNG DẪN (BẠN CHỈ CẦN PASTE ĐƯỜNG DẪN VÀO ĐÂY) 👇
# ==============================================================================

# 1. FILE ĐẦU VÀO VÀ ĐẦU RA
CONFIG = {
    # Đường dẫn file CSV chứa điểm ngập (Training data)
    'csv_file': r"../data/raw/FloodMarks1999.csv",

    # Đường dẫn file Shapefile (.shp) ranh giới Huế (để cắt kết quả)
    'shapefile': r"../data/spatial/Hue shapefile\hue.shp",

    # Nơi muốn lưu file kết quả dự báo (.tif) - Độ phân giải 30m
    'output_file': r"../results/rfkq\KetQua_Hue_30m.tif",
    
    # File bản đồ phân loại cấp độ ngập (0-5)
    'output_classified': r"../results/rfkq\KetQua_Hue_30m_PhanLoai.tif",
    
    # File chú giải bản đồ phân loại
    'output_legend': r"../results/rfkq\Flood_Classification_Legend.png",
    
    # File CSV chứa toàn bộ điểm dự báo (5 triệu điểm)
    'output_csv': r"../results/rfkq\KetQua_Hue_30m_FullData.csv",
    
    # File báo cáo đánh giá mô hình
    'evaluation_report': r"../results/rfkq\Model_Evaluation_Report.txt",
    
    # Biểu đồ feature importance
    'feature_importance_plot': r"../results/rfkq\Feature_Importance.png",
    
    # Biểu đồ scatter plot (Predicted vs Actual)
    'scatter_plot': r"../results/rfkq\Predicted_vs_Actual.png",
}

# 2. ĐƯỜNG DẪN CÁC FILE TIFF (RASTER)
TIFF_PATHS = {
    # --- Nhóm Địa hình (Quan trọng: dsm_bathy dùng làm khung tham chiếu) ---
    'dsm_bathy':   r"../data/raster\DEM_ThuaThienHue.tif",
    'slope':       r"../data/raster\Slope_ThuaThienHue.tif",
    'aspect':      r"../data/raster\Aspect_ThuaThienHue.tif",
    'curvature':   r"../data/raster\Curvature_ThuaThienHue.tif",
    'hillshade':   r"../data/raster\Hillshade_ThuaThienHue.tif",
    'roughness':   r"../data/raster\Roughness_ThuaThienHue.tif",
    'spi':         r"../data/raster\spi_10m.tif",
    'tpi':         r"../data/raster\TPI_ThuaThienHue.tif",
    'twi':         r"../data/raster\twi_10m.tif",
    
    # --- Nhóm Thủy văn ---
    'flow_accum':  r"../data/raster\flow_accum_10m.tif",
    'hand30_100':  r"../data/raster\HAND_30_100_ThuaThienHue.tif",
    'distance_r':  r"../data/raster\distance_r final.tif",
    'density_ri':  r"../data/raster\density_r.tif",
    'gsw_occure':  r"../data/raster\gsw.tif",
    
    # --- Nhóm Mưa & Lớp phủ ---
    'precip_s_1':  r"../data/raster\Hue_Rainfall_Total_1999_10mT12.tif",
    'VLUCD_L2_3':  r"../data/raster\lulc hue.tif",
}

# 3. CẤU HÌNH KHÁC
TARGET_COL = 'Flood_1999'  # Tên cột mục tiêu trong CSV
NODATA_VAL = -9999         # Giá trị không có dữ liệu
TARGET_RESOLUTION = 30     # Độ phân giải đầu ra (30m)

# ==============================================================================
# 🛑 HẾT PHẦN CẤU HÌNH
# ==============================================================================

def extract_values_at_points(tif_path, lons, lats):
    """Lấy giá trị pixel từ đường dẫn tuyệt đối"""
    if not tif_path or not os.path.exists(tif_path):
        print(f"⚠️  Bỏ qua: Không tìm thấy file {os.path.basename(tif_path) if tif_path else 'EMPTY'}")
        return np.zeros(len(lons))
    
    try:
        with rasterio.open(tif_path) as src:
            coords = list(zip(lons, lats))
            vals = [x[0] for x in src.sample(coords)]
            return np.array(vals)
    except Exception as e:
        print(f"❌ Lỗi đọc file {tif_path}: {e}")
        return np.zeros(len(lons))

def clip_raster_by_shapefile(input_tif, shapefile_path, output_tif):
    """Cắt file kết quả theo ranh giới Shapefile"""
    print("\n✂️  Đang thực hiện cắt bản đồ theo Shapefile...")
    
    try:
        gdf = gpd.read_file(shapefile_path)
        
        with rasterio.open(input_tif) as src:
            if str(gdf.crs) != str(src.crs):
                print("⚠️  Hệ tọa độ không khớp! Đang chuyển Shapefile về cùng hệ với Raster...")
                gdf = gdf.to_crs(src.crs)
            
            shapes = [feature["geometry"] for _, feature in gdf.iterrows()]
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, nodata=NODATA_VAL)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": NODATA_VAL,
            "compress": "lzw"
        })

        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(out_image)
            
        print("✅ Đã cắt bản đồ thành công!")
        
    except Exception as e:
        print(f"❌ Lỗi khi cắt Shapefile: {e}")

def classify_flood_depth(depth):
    """
    Phân loại độ sâu ngập lụt thành 6 cấp độ
    
    Tham khảo tiêu chuẩn quốc tế và Việt Nam:
    - Cấp 0: Không ngập (0m)
    - Cấp 1: Ngập nhẹ (0-0.5m) - Nước lên mắt cá chân
    - Cấp 2: Ngập trung bình (0.5-1.0m) - Nước lên đầu gối
    - Cấp 3: Ngập nặng (1.0-1.5m) - Nước lên thắt lưng
    - Cấp 4: Ngập rất nặng (1.5-2.0m) - Nước ngập gần đầu người
    - Cấp 5: Ngập đặc biệt nghiêm trọng (>2.0m) - Nước vượt đầu người
    
    Args:
        depth: Độ sâu ngập (m) - có thể là scalar hoặc array
    
    Returns:
        Cấp độ ngập (0-5)
    """
    if isinstance(depth, (int, float)):
        # Xử lý single value
        if depth <= 0:
            return 0
        elif depth <= 0.5:
            return 1
        elif depth <= 1.0:
            return 2
        elif depth <= 1.5:
            return 3
        elif depth <= 2.0:
            return 4
        else:
            return 5
    else:
        # Xử lý array
        classified = np.zeros_like(depth, dtype=np.uint8)
        classified[depth > 0] = 1      # Ngập nhẹ
        classified[depth > 0.5] = 2    # Ngập trung bình
        classified[depth > 1.0] = 3    # Ngập nặng
        classified[depth > 1.5] = 4    # Ngập rất nặng
        classified[depth > 2.0] = 5    # Ngập đặc biệt nghiêm trọng
        return classified

def create_flood_classification_map(depth_raster_path, output_classified_path, output_legend_path):
    """
    Tạo bản đồ phân loại cấp độ ngập lụt từ bản đồ độ sâu
    
    Args:
        depth_raster_path: Đường dẫn file raster độ sâu ngập
        output_classified_path: Đường dẫn file output phân loại
        output_legend_path: Đường dẫn file ảnh chú giải
    """
    print("\n🗺️  Đang tạo bản đồ phân loại cấp độ ngập lụt...")
    
    with rasterio.open(depth_raster_path) as src:
        depth_data = src.read(1)
        meta = src.meta.copy()
        
        # Phân loại
        classified_data = np.zeros_like(depth_data, dtype=np.uint8)
        mask = depth_data != NODATA_VAL
        
        classified_data[mask] = classify_flood_depth(depth_data[mask])
        classified_data[~mask] = 255  # NoData = 255
        
        # Thống kê
        unique, counts = np.unique(classified_data[mask], return_counts=True)
        total_pixels = np.sum(counts)
        
        print("\n   📊 THỐNG KÊ PHÂN LOẠI:")
        class_names = [
            "Cấp 0: Không ngập (0m)",
            "Cấp 1: Ngập nhẹ (0-0.5m)",
            "Cấp 2: Ngập trung bình (0.5-1.0m)",
            "Cấp 3: Ngập nặng (1.0-1.5m)",
            "Cấp 4: Ngập rất nặng (1.5-2.0m)",
            "Cấp 5: Ngập đặc biệt nghiêm trọng (>2.0m)"
        ]
        
        for cls, count in zip(unique, counts):
            if cls < len(class_names):
                percentage = (count / total_pixels) * 100
                print(f"   {class_names[int(cls)]}: {count:,} pixels ({percentage:.2f}%)")
        
        # Cập nhật metadata
        meta.update({
            'dtype': 'uint8',
            'nodata': 255,
            'compress': 'lzw'
        })
        
        # Ghi file
        with rasterio.open(output_classified_path, 'w', **meta) as dst:
            dst.write(classified_data, 1)
            
            # Thêm color map
            colormap = {
                0: (255, 255, 255),    # Trắng - Không ngập
                1: (255, 255, 0),      # Vàng - Ngập nhẹ
                2: (255, 200, 0),      # Vàng cam - Ngập trung bình
                3: (255, 150, 0),      # Cam - Ngập nặng
                4: (255, 100, 0),      # Cam đỏ - Ngập rất nặng
                5: (255, 0, 0),        # Đỏ - Ngập đặc biệt nghiêm trọng
                255: (0, 0, 0)         # Đen - NoData
            }
            dst.write_colormap(1, colormap)
    
    print(f"   ✅ Đã lưu bản đồ phân loại: {output_classified_path}")
    
    # Tạo chú giải (legend)
    create_flood_legend(output_legend_path, class_names)
    
    return classified_data

def create_flood_legend(output_path, class_names):
    """Tạo ảnh chú giải cho bản đồ phân loại"""
    print(f"   -> Đang tạo chú giải bản đồ...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = [
        '#FFFFFF',  # Trắng - Không ngập
        '#FFFF00',  # Vàng - Ngập nhẹ
        '#FFC800',  # Vàng cam
        '#FF9600',  # Cam
        '#FF6400',  # Cam đỏ
        '#FF0000'   # Đỏ
    ]
    
    # Vẽ legend
    for i, (color, name) in enumerate(zip(colors, class_names)):
        ax.barh(i, 1, color=color, edgecolor='black', linewidth=2)
        ax.text(1.1, i, name, va='center', fontsize=11, weight='bold')
    
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.5, len(class_names) - 0.5)
    ax.axis('off')
    ax.set_title('PHÂN LOẠI CẤP ĐỘ NGẬP LỤT\nTỉnh Thừa Thiên Huế - 1999', 
                 fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Đã lưu chú giải: {output_path}")

def evaluate_model(y_true, y_pred, feature_names, feature_importance):
    """Đánh giá mô hình và tạo báo cáo"""
    print("\n📊 ĐÁNH GIÁ MÔ HÌNH:")
    
    # Tính toán các metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Tính MAPE (Mean Absolute Percentage Error)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0
    
    print(f"   • RMSE: {rmse:.4f} m")
    print(f"   • MAE: {mae:.4f} m")
    print(f"   • R²: {r2:.4f}")
    print(f"   • MAPE: {mape:.2f}%")
    
    # Lưu báo cáo
    report_path = CONFIG['evaluation_report']
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST - DỰ BÁO NGẬP LỤT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. THÔNG SỐ MÔ HÌNH:\n")
        f.write(f"   - Số lượng cây: 1000\n")
        f.write(f"   - Max depth: 20\n")
        f.write(f"   - Min samples split: 5\n")
        f.write(f"   - Min samples leaf: 2\n")
        f.write(f"   - Max features: sqrt\n")
        f.write(f"   - Số lượng features: {len(feature_names)}\n")
        f.write(f"   - Số điểm huấn luyện: {len(y_true)}\n\n")
        
        f.write("2. CHỈ SỐ ĐÁNH GIÁ:\n")
        f.write(f"   - RMSE (Root Mean Square Error): {rmse:.4f} m\n")
        f.write(f"   - MAE (Mean Absolute Error): {mae:.4f} m\n")
        f.write(f"   - R² (Coefficient of Determination): {r2:.4f}\n")
        f.write(f"   - MAPE (Mean Absolute Percentage Error): {mape:.2f}%\n\n")
        
        f.write("3. FEATURE IMPORTANCE (Top 10):\n")
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        for idx, row in feature_imp_df.head(10).iterrows():
            f.write(f"   {row['Feature']:20s}: {row['Importance']:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"   ✅ Đã lưu báo cáo tại: {report_path}")
    
    # Vẽ biểu đồ Feature Importance
    plt.figure(figsize=(10, 8))
    feature_imp_df = feature_imp_df.sort_values('Importance', ascending=True).tail(15)
    plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance - Random Forest Model')
    plt.tight_layout()
    plt.savefig(CONFIG['feature_importance_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu biểu đồ Feature Importance tại: {CONFIG['feature_importance_plot']}")
    
    # Vẽ biểu đồ Predicted vs Actual
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Flood Depth (m)')
    plt.ylabel('Predicted Flood Depth (m)')
    plt.title(f'Predicted vs Actual (R² = {r2:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['scatter_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu biểu đồ Predicted vs Actual tại: {CONFIG['scatter_plot']}")
    
    return rmse, mae, r2, mape

def main():
    print("🌊 CHƯƠNG TRÌNH DỰ BÁO NGẬP LỤT - XUẤT KẾT QUẢ 30M")
    
    # Tạo thư mục output
    output_dir = os.path.dirname(CONFIG['output_file'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Tạo thư mục: {output_dir}")
    
    # 1. Load dữ liệu Train
    if not os.path.exists(CONFIG['csv_file']):
        print("❌ Lỗi: Không tìm thấy file CSV!")
        return

    print("\n1️⃣  Đang xử lý dữ liệu huấn luyện...")
    df = pd.read_csv(CONFIG['csv_file'])
    
    valid_features = {k: v for k, v in TIFF_PATHS.items() if v and os.path.exists(v)}
    feature_names = list(valid_features.keys())
    
    print(f"   -> Tìm thấy {len(valid_features)} file Tiff hợp lệ.")
    
    # Sampling dữ liệu
    for feat, path in tqdm(valid_features.items(), desc="   Sampling dữ liệu"):
        df[feat] = extract_values_at_points(path, df['lon_new'], df['lat_new'])
        
    df_clean = df.dropna(subset=[TARGET_COL])
    
    # HIỆU CHỈNH: Tạo thêm các feature tương tác để giảm phụ thuộc vào DEM
    print("   -> Đang tạo feature tương tác nâng cao...")
    
    # Nhóm 1: Tương tác DEM với các yếu tố khác
    if 'dsm_bathy' in feature_names and 'distance_r' in feature_names:
        df_clean['dem_x_dist'] = df_clean['dsm_bathy'] * df_clean['distance_r']
        feature_names.append('dem_x_dist')
    
    if 'dsm_bathy' in feature_names and 'slope' in feature_names:
        df_clean['dem_x_slope'] = df_clean['dsm_bathy'] * df_clean['slope']
        feature_names.append('dem_x_slope')
    
    if 'dsm_bathy' in feature_names and 'twi' in feature_names:
        df_clean['dem_x_twi'] = df_clean['dsm_bathy'] * df_clean['twi']
        feature_names.append('dem_x_twi')
    
    # Nhóm 2: Tương tác mưa với địa hình
    if 'precip_s_1' in feature_names and 'slope' in feature_names:
        df_clean['rain_x_slope'] = df_clean['precip_s_1'] * df_clean['slope']
        feature_names.append('rain_x_slope')
    
    if 'precip_s_1' in feature_names and 'twi' in feature_names:
        df_clean['rain_x_twi'] = df_clean['precip_s_1'] * df_clean['twi']
        feature_names.append('rain_x_twi')
    
    # Nhóm 3: Tương tác nâng cao (máy mạnh nên thêm nhiều features)
    if 'flow_accum' in feature_names and 'slope' in feature_names:
        df_clean['flow_x_slope'] = df_clean['flow_accum'] * df_clean['slope']
        feature_names.append('flow_x_slope')
    
    if 'distance_r' in feature_names and 'twi' in feature_names:
        df_clean['dist_x_twi'] = df_clean['distance_r'] * df_clean['twi']
        feature_names.append('dist_x_twi')
    
    if 'precip_s_1' in feature_names and 'flow_accum' in feature_names:
        df_clean['rain_x_flow'] = df_clean['precip_s_1'] * df_clean['flow_accum']
        feature_names.append('rain_x_flow')
    
    # Nhóm 4: Features bình phương (để bắt phi tuyến)
    if 'dsm_bathy' in feature_names:
        df_clean['dem_squared'] = df_clean['dsm_bathy'] ** 2
        feature_names.append('dem_squared')
    
    if 'distance_r' in feature_names:
        df_clean['dist_squared'] = df_clean['distance_r'] ** 2
        feature_names.append('dist_squared')
    
    print(f"   -> Tổng số features sau khi tạo tương tác: {len(feature_names)}")
    
    # 2. Train & Evaluate Model
    print("\n2️⃣  Đang huấn luyện và đánh giá Random Forest...")
    X = df_clean[feature_names]
    y = df_clean[TARGET_COL]
    
    # HIỆU CHỈNH: Chuẩn hóa dữ liệu để giảm ảnh hưởng của DEM
    from sklearn.preprocessing import StandardScaler
    
    print("   -> Đang chuẩn hóa dữ liệu (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # HIỆU CHỈNH: Tối ưu cho máy mạnh (32GB RAM, CPU nhiều cores)
    print("   -> Đang train model với tham số tối ưu cho máy mạnh...")
    rf = RandomForestRegressor(
        n_estimators=1000,          # Tăng lên 1000 cây cho độ chính xác tối đa
        max_depth=20,               # Tăng depth để model học được pattern phức tạp hơn
        min_samples_split=5,        # Giảm để model linh hoạt hơn
        min_samples_leaf=2,         # Giảm để tăng độ chi tiết
        max_features='sqrt',        # Giữ nguyên để tránh overfit
        max_samples=0.8,            # Bootstrap 80% data mỗi tree
        n_jobs=-1,                  # Dùng hết CPU cores
        random_state=42,
        verbose=1,                  # Hiển thị progress
        warm_start=False,           
        oob_score=True              # Tính Out-of-Bag score để đánh giá
    )
    rf.fit(X_train, y_train)
    
    print(f"   -> OOB Score: {rf.oob_score_:.4f}")  # Đánh giá thêm
    rf.fit(X_train, y_train)
    
    # Dự báo trên tập test
    y_pred = rf.predict(X_test)
    
    # Đánh giá model
    evaluate_model(y_test, y_pred, feature_names, rf.feature_importances_)

    # 3. Dự báo toàn bộ ảnh ở độ phân giải 30m
    print("\n3️⃣  Đang chạy dự báo ra file Raster 30m...")
    
    ref_path = TIFF_PATHS.get('dsm_bathy')
    if not ref_path or not os.path.exists(ref_path):
        print("❌ Lỗi: Không tìm thấy file dsm_bathy!")
        return

    # Mở file reference để lấy thông tin
    with rasterio.open(ref_path) as src_ref:
        print(f"   -> File gốc: {src_ref.width} x {src_ref.height} pixels")
        print(f"   -> CRS: {src_ref.crs}")
        print(f"   -> Pixel size gốc: {src_ref.res[0]:.10f} x {src_ref.res[1]:.10f} degrees")
        
        # QUAN TRỌNG: Với WGS84, 1 độ ở vĩ độ 16° ≈ 111km
        # 30m = 30/111000 ≈ 0.00027 độ
        degrees_per_30m = TARGET_RESOLUTION / 111000.0
        
        print(f"   -> 30m ≈ {degrees_per_30m:.10f} degrees")
        
        # Tính scale factor dựa trên degrees
        scale = degrees_per_30m / src_ref.res[0]
        
        print(f"   -> Scale factor: {scale:.4f}")
        
        # Tính kích thước mới
        new_width = int(src_ref.width / scale)
        new_height = int(src_ref.height / scale)
        
        print(f"   -> Kích thước mới: {new_width} x {new_height} pixels")
        print(f"   -> Tổng số pixel: {new_width * new_height:,} (~{new_width * new_height / 1e6:.2f}M điểm)")
        
        # Tạo transform mới với pixel size = degrees_per_30m
        from rasterio.transform import Affine
        old_transform = src_ref.transform
        new_transform = Affine(
            degrees_per_30m,    # pixel width (degrees)
            old_transform.b,    # rotation
            old_transform.c,    # x origin (top-left longitude)
            old_transform.d,    # rotation  
            -degrees_per_30m,   # pixel height (negative degrees for north-up)
            old_transform.f     # y origin (top-left latitude)
        )
        
        # Metadata cho output
        out_meta = src_ref.meta.copy()
        out_meta.update({
            'width': new_width,
            'height': new_height,
            'transform': new_transform,
            'count': 1,
            'dtype': 'float32',
            'nodata': NODATA_VAL,
            'compress': 'lzw'
        })
    
    # Mở tất cả file nguồn
    src_files = {}
    for feat, path in valid_features.items():
        src_files[feat] = rasterio.open(path)

    temp_output = CONFIG['output_file'].replace('.tif', '_temp.tif')
    all_data = []
    
    try:
        with rasterio.open(temp_output, 'w', **out_meta) as dst:
            # Tối ưu block size cho 32GB RAM - xử lý block lớn hơn
            block_size = 1024  # Tăng từ 256 lên 1024 (xử lý ~1M pixels/lần)
            
            for row_start in tqdm(range(0, new_height, block_size), desc="   Đang dự báo"):
                row_end = min(row_start + block_size, new_height)
                rows = row_end - row_start
                cols = new_width
                
                # Window cho ảnh 30m output
                window_30m = Window(0, row_start, cols, rows)
                
                # Window tương ứng trên ảnh 10m (nhân với scale)
                window_10m = Window(
                    col_off=0,
                    row_off=int(row_start * scale),
                    width=int(cols * scale),
                    height=int(rows * scale)
                )
                
                # Ma trận input
                X_block = np.zeros((rows * cols, len(valid_features)), dtype=np.float32)
                mask_valid = np.ones((rows * cols), dtype=bool)
                
                # Dictionary để lưu dữ liệu feature gốc
                feature_data = {}
                
                # Đọc và resample từng feature
                for i, feat in enumerate(valid_features.keys()):
                    try:
                        data = src_files[feat].read(
                            1,
                            window=window_10m,
                            out_shape=(rows, cols),
                            resampling=Resampling.bilinear,
                            boundless=True,
                            fill_value=0
                        ).flatten()
                        
                        X_block[:, i] = data
                        feature_data[feat] = data
                        
                        # Mask dựa trên DEM
                        if feat == 'dsm_bathy':
                            nodata = src_files[feat].nodata if src_files[feat].nodata is not None else -9999
                            mask_valid = (data > -100) & (data != nodata)
                    except Exception as e:
                        print(f"⚠️  Lỗi đọc {feat}: {e}")
                        X_block[:, i] = 0
                        feature_data[feat] = np.zeros(rows * cols)
                
                # Tạo features tương tác (giống như lúc train)
                interaction_features = []
                
                # Nhóm 1: DEM interactions
                if 'dsm_bathy' in feature_data and 'distance_r' in feature_data:
                    interaction_features.append(feature_data['dsm_bathy'] * feature_data['distance_r'])
                
                if 'dsm_bathy' in feature_data and 'slope' in feature_data:
                    interaction_features.append(feature_data['dsm_bathy'] * feature_data['slope'])
                
                if 'dsm_bathy' in feature_data and 'twi' in feature_data:
                    interaction_features.append(feature_data['dsm_bathy'] * feature_data['twi'])
                
                # Nhóm 2: Rain interactions
                if 'precip_s_1' in feature_data and 'slope' in feature_data:
                    interaction_features.append(feature_data['precip_s_1'] * feature_data['slope'])
                
                if 'precip_s_1' in feature_data and 'twi' in feature_data:
                    interaction_features.append(feature_data['precip_s_1'] * feature_data['twi'])
                
                # Nhóm 3: Advanced interactions
                if 'flow_accum' in feature_data and 'slope' in feature_data:
                    interaction_features.append(feature_data['flow_accum'] * feature_data['slope'])
                
                if 'distance_r' in feature_data and 'twi' in feature_data:
                    interaction_features.append(feature_data['distance_r'] * feature_data['twi'])
                
                if 'precip_s_1' in feature_data and 'flow_accum' in feature_data:
                    interaction_features.append(feature_data['precip_s_1'] * feature_data['flow_accum'])
                
                # Nhóm 4: Squared features
                if 'dsm_bathy' in feature_data:
                    interaction_features.append(feature_data['dsm_bathy'] ** 2)
                
                if 'distance_r' in feature_data:
                    interaction_features.append(feature_data['distance_r'] ** 2)
                
                # Gộp features gốc và features tương tác
                if interaction_features:
                    X_block_full = np.column_stack([X_block] + interaction_features)
                else:
                    X_block_full = X_block
                
                # Dự báo
                y_block = np.full(rows * cols, NODATA_VAL, dtype=np.float32)
                if np.any(mask_valid):
                    X_valid = np.nan_to_num(X_block_full[mask_valid])
                    # QUAN TRỌNG: Phải scale dữ liệu giống như lúc train
                    X_valid_scaled = scaler.transform(X_valid)
                    y_block[mask_valid] = rf.predict(X_valid_scaled)
                
                # Ghi vào raster
                dst.write(y_block.reshape(rows, cols), 1, window=window_30m)
                
                # Lưu cho CSV (chỉ điểm hợp lệ)
                if np.any(mask_valid):
                    # Tạo lưới tọa độ
                    row_idx, col_idx = np.meshgrid(
                        np.arange(row_start, row_end),
                        np.arange(0, cols),
                        indexing='ij'
                    )
                    
                    # Chuyển pixel -> tọa độ địa lý
                    xs, ys = rasterio.transform.xy(
                        new_transform,
                        row_idx.flatten()[mask_valid],
                        col_idx.flatten()[mask_valid]
                    )
                    
                    block_df = pd.DataFrame(X_block[mask_valid], columns=list(valid_features.keys()))
                    block_df['lon'] = xs
                    block_df['lat'] = ys
                    block_df['predicted_depth'] = y_block[mask_valid]
                    all_data.append(block_df)
                    
    finally:
        for src in src_files.values():
            src.close()

    # 4. Cắt theo Shapefile
    if CONFIG['shapefile'] and os.path.exists(CONFIG['shapefile']):
        clip_raster_by_shapefile(temp_output, CONFIG['shapefile'], CONFIG['output_file'])
        if os.path.exists(temp_output):
            os.remove(temp_output)
    else:
        if os.path.exists(temp_output):
            os.rename(temp_output, CONFIG['output_file'])
        print("⚠️  Không tìm thấy Shapefile, xuất ra kết quả hình chữ nhật gốc.")

    # 5. Tạo bản đồ phân loại cấp độ ngập lụt
    print("\n5️⃣  Đang tạo bản đồ phân loại cấp độ ngập lụt...")
    create_flood_classification_map(
        CONFIG['output_file'], 
        CONFIG['output_classified'],
        CONFIG['output_legend']
    )

    # 6. Xuất CSV toàn bộ điểm
    print("\n4️⃣  Đang xuất file CSV chứa toàn bộ điểm dự báo...")
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df.to_csv(CONFIG['output_csv'], index=False)
        print(f"   ✅ Đã xuất {len(full_df):,} điểm vào file: {CONFIG['output_csv']}")
    else:
        print("   ⚠️ Không có dữ liệu để xuất CSV!")

    print(f"\n🎉 HOÀN TẤT!")
    print(f"   📁 File Raster độ sâu 30m: {CONFIG['output_file']}")
    print(f"   📁 File Raster phân loại 6 cấp độ: {CONFIG['output_classified']}")
    print(f"   📁 File chú giải bản đồ: {CONFIG['output_legend']}")
    print(f"   📁 File CSV Full Data: {CONFIG['output_csv']}")
    print(f"   📁 Báo cáo đánh giá: {CONFIG['evaluation_report']}")

if __name__ == "__main__":
    main()
