import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.utils import *
import numpy as np
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
    'output_file': r"../results/rfkq\XGB_KetQua_Hue_30m.tif",
    
    # File bản đồ phân loại cấp độ ngập (0-5)
    'output_classified': r"../results/rfkq\XGB_KetQua_Hue_30m_PhanLoai.tif",
    
    # File chú giải bản đồ phân loại
    'output_legend': r"../results/rfkq\XGB_Flood_Classification_Legend.png",
    
    # File CSV chứa toàn bộ điểm dự báo (5 triệu điểm)
    'output_csv': r"../results/rfkq\XGB_KetQua_Hue_30m_FullData.csv",
    
    # File báo cáo đánh giá mô hình
    'evaluation_report': r"../results/rfkq\XGB_Model_Evaluation_Report.txt",
    
    # Biểu đồ feature importance
    'feature_importance_plot': r"../results/rfkq\XGB_Feature_Importance.png",
    
    # Biểu đồ scatter plot (Predicted vs Actual)
    'scatter_plot': r"../results/rfkq\XGB_Predicted_vs_Actual.png",
    
    # File lưu model (để tái sử dụng)
    'model_file': r"../results/rfkq\XGB_model.json",
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





def create_flood_legend(output_path, class_names):
    """Tạo chú giải bản đồ"""
    print(f"   -> Đang tạo chú giải bản đồ...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#FFFFFF', '#FFFF00', '#FFC800', '#FF9600', '#FF6400', '#FF0000']
    
    for i, (color, name) in enumerate(zip(colors, class_names)):
        ax.barh(i, 1, color=color, edgecolor='black', linewidth=2)
        ax.text(1.1, i, name, va='center', fontsize=11, weight='bold')
    
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.5, len(class_names) - 0.5)
    ax.axis('off')
    ax.set_title('PHÂN LOẠI CẤP ĐỘ NGẬP LỤT - XGBoost\nTỉnh Thừa Thiên Huế - 1999', 
                 fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Đã lưu chú giải: {output_path}")

def evaluate_model(y_true, y_pred, feature_names, feature_importance, model_type="XGBoost"):
    """Đánh giá mô hình"""
    print(f"\n📊 ĐÁNH GIÁ MÔ HÌNH {model_type}:")
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0
    
    print(f"   • RMSE: {rmse:.4f} m")
    print(f"   • MAE: {mae:.4f} m")
    print(f"   • R²: {r2:.4f}")
    print(f"   • MAPE: {mape:.2f}%")
    
    report_path = CONFIG['evaluation_report']
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"BÁO CÁO ĐÁNH GIÁ MÔ HÌNH {model_type} - DỰ BÁO NGẬP LỤT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"1. THÔNG SỐ MÔ HÌNH {model_type}:\n")
        f.write(f"   - Device: GPU (RTX 3080)\n")
        f.write(f"   - Số lượng features: {len(feature_names)}\n")
        f.write(f"   - Số điểm huấn luyện: {len(y_true)}\n\n")
        f.write("2. CHỈ SỐ ĐÁNH GIÁ:\n")
        f.write(f"   - RMSE: {rmse:.4f} m\n")
        f.write(f"   - MAE: {mae:.4f} m\n")
        f.write(f"   - R²: {r2:.4f}\n")
        f.write(f"   - MAPE: {mape:.2f}%\n\n")
        f.write("3. FEATURE IMPORTANCE (Top 10):\n")
        
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        for idx, row in feature_imp_df.head(10).iterrows():
            f.write(f"   {row['Feature']:20s}: {row['Importance']:.4f}\n")
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"   ✅ Đã lưu báo cáo tại: {report_path}")
    
    # Feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = feature_imp_df.sort_values('Importance', ascending=True).tail(15)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importance - {model_type} Model')
    plt.tight_layout()
    plt.savefig(CONFIG['feature_importance_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu Feature Importance: {CONFIG['feature_importance_plot']}")
    
    # Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Flood Depth (m)')
    plt.ylabel('Predicted Flood Depth (m)')
    plt.title(f'Predicted vs Actual (R² = {r2:.4f}) - {model_type}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['scatter_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu Scatter Plot: {CONFIG['scatter_plot']}")
    
    return rmse, mae, r2, mape

def main():
    print("🌊 CHƯƠNG TRÌNH DỰ BÁO NGẬP LỤT - XGBoost GPU (RTX 3080)")
    
    # Tạo thư mục output
    output_dir = os.path.dirname(CONFIG['output_file'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Tạo thư mục: {output_dir}")
    
    # 1. Load dữ liệu
    if not os.path.exists(CONFIG['csv_file']):
        print("❌ Lỗi: Không tìm thấy file CSV!")
        return

    print("\n1️⃣  Đang xử lý dữ liệu huấn luyện...")
    df = pd.read_csv(CONFIG['csv_file'])
    
    valid_features = {k: v for k, v in TIFF_PATHS.items() if v and os.path.exists(v)}
    feature_names = list(valid_features.keys())
    
    print(f"   -> Tìm thấy {len(valid_features)} file Tiff hợp lệ.")
    
    # Sampling
    for feat, path in tqdm(valid_features.items(), desc="   Sampling dữ liệu"):
        df[feat] = extract_values_at_points(path, df['lon_new'], df['lat_new'])
        
    df_clean = df.dropna(subset=[TARGET_COL])
    
    # Feature engineering
    print("   -> Đang tạo feature tương tác nâng cao...")
    
    if 'dsm_bathy' in feature_names and 'distance_r' in feature_names:
        df_clean['dem_x_dist'] = df_clean['dsm_bathy'] * df_clean['distance_r']
        feature_names.append('dem_x_dist')
    
    if 'dsm_bathy' in feature_names and 'slope' in feature_names:
        df_clean['dem_x_slope'] = df_clean['dsm_bathy'] * df_clean['slope']
        feature_names.append('dem_x_slope')
    
    if 'dsm_bathy' in feature_names and 'twi' in feature_names:
        df_clean['dem_x_twi'] = df_clean['dsm_bathy'] * df_clean['twi']
        feature_names.append('dem_x_twi')
    
    if 'precip_s_1' in feature_names and 'slope' in feature_names:
        df_clean['rain_x_slope'] = df_clean['precip_s_1'] * df_clean['slope']
        feature_names.append('rain_x_slope')
    
    if 'precip_s_1' in feature_names and 'twi' in feature_names:
        df_clean['rain_x_twi'] = df_clean['precip_s_1'] * df_clean['twi']
        feature_names.append('rain_x_twi')
    
    if 'flow_accum' in feature_names and 'slope' in feature_names:
        df_clean['flow_x_slope'] = df_clean['flow_accum'] * df_clean['slope']
        feature_names.append('flow_x_slope')
    
    if 'distance_r' in feature_names and 'twi' in feature_names:
        df_clean['dist_x_twi'] = df_clean['distance_r'] * df_clean['twi']
        feature_names.append('dist_x_twi')
    
    if 'precip_s_1' in feature_names and 'flow_accum' in feature_names:
        df_clean['rain_x_flow'] = df_clean['precip_s_1'] * df_clean['flow_accum']
        feature_names.append('rain_x_flow')
    
    if 'dsm_bathy' in feature_names:
        df_clean['dem_squared'] = df_clean['dsm_bathy'] ** 2
        feature_names.append('dem_squared')
    
    if 'distance_r' in feature_names:
        df_clean['dist_squared'] = df_clean['distance_r'] ** 2
        feature_names.append('dist_squared')
    
    print(f"   -> Tổng số features: {len(feature_names)}")
    
    # 2. Train XGBoost với GPU
    print("\n2️⃣  Đang huấn luyện XGBoost trên GPU (RTX 3080)...")
    X = df_clean[feature_names]
    y = df_clean[TARGET_COL]
    
    # Chuẩn hóa
    print("   -> Chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # XGBoost parameters - TỐI ƯU CHO RTX 3080
    print("   -> Khởi tạo XGBoost với GPU acceleration...")
    params = {
        'device': 'cuda',              # SỬ DỤNG GPU!
        'tree_method': 'hist',         # GPU-optimized histogram method
        'max_depth': 12,               # Depth tối ưu
        'learning_rate': 0.05,         # Learning rate vừa phải
        'n_estimators': 2000,          # Nhiều trees (GPU xử lý nhanh)
        'subsample': 0.8,              # Bootstrap sampling
        'colsample_bytree': 0.8,       # Feature sampling
        'min_child_weight': 3,
        'gamma': 0.1,                  # Regularization
        'reg_alpha': 0.1,              # L1 regularization
        'reg_lambda': 1.0,             # L2 regularization
        'random_state': 42,
        'n_jobs': -1,                  # Sử dụng tất cả CPU cores
        'verbosity': 1
    }
    
    # Train
    model = xgb.XGBRegressor(**params)
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=100
    )
    
    print(f"\n   ✅ Training hoàn tất!")
    
    # Dự báo
    y_pred = model.predict(X_test)
    
    # Đánh giá
    evaluate_model(y_test, y_pred, feature_names, model.feature_importances_, "XGBoost-GPU")
    
    # Lưu model
    print(f"\n   -> Đang lưu model...")
    model.save_model(CONFIG['model_file'])
    print(f"   ✅ Đã lưu model: {CONFIG['model_file']}")
    
    # 3. Dự báo toàn bộ ảnh
    print("\n3️⃣  Đang chạy dự báo ra file Raster 30m với XGBoost-GPU...")
    
    ref_path = TIFF_PATHS.get('dsm_bathy')
    if not ref_path or not os.path.exists(ref_path):
        print("❌ Lỗi: Không tìm thấy file dsm_bathy!")
        return

    with rasterio.open(ref_path) as src_ref:
        print(f"   -> File gốc: {src_ref.width} x {src_ref.height} pixels")
        print(f"   -> CRS: {src_ref.crs}")
        
        degrees_per_30m = TARGET_RESOLUTION / 111000.0
        scale = degrees_per_30m / src_ref.res[0]
        
        new_width = int(src_ref.width / scale)
        new_height = int(src_ref.height / scale)
        
        print(f"   -> Kích thước output: {new_width} x {new_height} pixels (~{new_width*new_height/1e6:.2f}M điểm)")
        
        from rasterio.transform import Affine
        old_transform = src_ref.transform
        new_transform = Affine(
            degrees_per_30m, old_transform.b, old_transform.c,
            old_transform.d, -degrees_per_30m, old_transform.f
        )
        
        out_meta = src_ref.meta.copy()
        out_meta.update({
            'width': new_width, 'height': new_height, 'transform': new_transform,
            'count': 1, 'dtype': 'float32', 'nodata': NODATA_VAL, 'compress': 'lzw'
        })
    
    src_files = {}
    for feat, path in valid_features.items():
        src_files[feat] = rasterio.open(path)

    temp_output = CONFIG['output_file'].replace('.tif', '_temp.tif')
    all_data = []
    
    try:
        with rasterio.open(temp_output, 'w', **out_meta) as dst:
            block_size = 1024  # Large blocks for fast processing
            
            for row_start in tqdm(range(0, new_height, block_size), desc="   Đang dự báo với GPU"):
                row_end = min(row_start + block_size, new_height)
                rows = row_end - row_start
                cols = new_width
                
                window_30m = Window(0, row_start, cols, rows)
                window_10m = Window(0, int(row_start * scale), int(cols * scale), int(rows * scale))
                
                X_block = np.zeros((rows * cols, len(valid_features)), dtype=np.float32)
                mask_valid = np.ones((rows * cols), dtype=bool)
                feature_data = {}
                
                for i, feat in enumerate(valid_features.keys()):
                    try:
                        data = src_files[feat].read(
                            1, window=window_10m, out_shape=(rows, cols),
                            resampling=Resampling.bilinear, boundless=True, fill_value=0
                        ).flatten()
                        
                        X_block[:, i] = data
                        feature_data[feat] = data
                        
                        if feat == 'dsm_bathy':
                            nodata = src_files[feat].nodata if src_files[feat].nodata is not None else -9999
                            mask_valid = (data > -100) & (data != nodata)
                    except Exception as e:
                        X_block[:, i] = 0
                        feature_data[feat] = np.zeros(rows * cols)
                
                # Tạo interaction features
                interaction_features = []
                if 'dsm_bathy' in feature_data and 'distance_r' in feature_data:
                    interaction_features.append(feature_data['dsm_bathy'] * feature_data['distance_r'])
                if 'dsm_bathy' in feature_data and 'slope' in feature_data:
                    interaction_features.append(feature_data['dsm_bathy'] * feature_data['slope'])
                if 'dsm_bathy' in feature_data and 'twi' in feature_data:
                    interaction_features.append(feature_data['dsm_bathy'] * feature_data['twi'])
                if 'precip_s_1' in feature_data and 'slope' in feature_data:
                    interaction_features.append(feature_data['precip_s_1'] * feature_data['slope'])
                if 'precip_s_1' in feature_data and 'twi' in feature_data:
                    interaction_features.append(feature_data['precip_s_1'] * feature_data['twi'])
                if 'flow_accum' in feature_data and 'slope' in feature_data:
                    interaction_features.append(feature_data['flow_accum'] * feature_data['slope'])
                if 'distance_r' in feature_data and 'twi' in feature_data:
                    interaction_features.append(feature_data['distance_r'] * feature_data['twi'])
                if 'precip_s_1' in feature_data and 'flow_accum' in feature_data:
                    interaction_features.append(feature_data['precip_s_1'] * feature_data['flow_accum'])
                if 'dsm_bathy' in feature_data:
                    interaction_features.append(feature_data['dsm_bathy'] ** 2)
                if 'distance_r' in feature_data:
                    interaction_features.append(feature_data['distance_r'] ** 2)
                
                if interaction_features:
                    X_block_full = np.column_stack([X_block] + interaction_features)
                else:
                    X_block_full = X_block
                
                # Dự báo với GPU
                y_block = np.full(rows * cols, NODATA_VAL, dtype=np.float32)
                if np.any(mask_valid):
                    X_valid = np.nan_to_num(X_block_full[mask_valid])
                    X_valid_scaled = scaler.transform(X_valid)
                    y_block[mask_valid] = model.predict(X_valid_scaled)
                
                dst.write(y_block.reshape(rows, cols), 1, window=window_30m)
                
                if np.any(mask_valid):
                    row_idx, col_idx = np.meshgrid(
                        np.arange(row_start, row_end), np.arange(0, cols), indexing='ij'
                    )
                    xs, ys = rasterio.transform.xy(
                        new_transform, row_idx.flatten()[mask_valid], col_idx.flatten()[mask_valid]
                    )
                    block_df = pd.DataFrame(X_block[mask_valid], columns=list(valid_features.keys()))
                    block_df['lon'] = xs
                    block_df['lat'] = ys
                    block_df['predicted_depth'] = y_block[mask_valid]
                    all_data.append(block_df)
                    
    finally:
        for src in src_files.values():
            src.close()

    # 4. Clip shapefile
    if CONFIG['shapefile'] and os.path.exists(CONFIG['shapefile']):
        clip_raster_by_shapefile(temp_output, CONFIG['shapefile'], CONFIG['output_file'])
        if os.path.exists(temp_output):
            os.remove(temp_output)
    else:
        if os.path.exists(temp_output):
            os.rename(temp_output, CONFIG['output_file'])

    # 5. Phân loại
    print("\n4️⃣  Đang tạo bản đồ phân loại...")
    create_flood_classification_map(CONFIG['output_file'], CONFIG['output_classified'], CONFIG['output_legend'])

    # 6. Export CSV
    print("\n5️⃣  Đang xuất CSV...")
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df.to_csv(CONFIG['output_csv'], index=False)
        print(f"   ✅ Đã xuất {len(full_df):,} điểm")

    print(f"\n🎉 HOÀN TẤT - XGBoost GPU!")
    print(f"   📁 Raster độ sâu: {CONFIG['output_file']}")
    print(f"   📁 Raster phân loại: {CONFIG['output_classified']}")
    print(f"   📁 Chú giải: {CONFIG['output_legend']}")
    print(f"   📁 CSV: {CONFIG['output_csv']}")
    print(f"   📁 Model: {CONFIG['model_file']}")
    print(f"   📁 Báo cáo: {CONFIG['evaluation_report']}")

if __name__ == "__main__":
    main()
