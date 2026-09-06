"""
DEEP NEURAL NETWORK - PHIÊN BẢN ỔN ĐỊNH
Khắc phục: NaN values, numerical stability
"""

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
from sklearn.preprocessing import RobustScaler  # Thay StandardScaler
import os
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam

# Cấu hình
print("=" * 70)
print("🔍 KIỂM TRA GPU:")
print("=" * 70)
print(f"TensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU: {len(gpus)} device(s)")
    except:
        print("⚠️  GPU config failed")
else:
    print("⚠️  CPU only")
print("=" * 70 + "\n")

CONFIG = {
    'csv_file': r"../data/raw/FloodMarks1999.csv",
    'shapefile': r"../data/spatial/Hue shapefile\hue.shp",
    'output_file': r"../results/rfkq\DNN_Stable_KetQua_Hue_30m.tif",
    'output_classified': r"../results/rfkq\DNN_Stable_KetQua_Hue_30m_PhanLoai.tif",
    'output_legend': r"../results/rfkq\DNN_Stable_Legend.png",
    'output_csv': r"../results/rfkq\DNN_Stable_FullData.csv",
    'evaluation_report': r"../results/rfkq\DNN_Stable_Report.txt",
    'training_history_plot': r"../results/rfkq\DNN_Stable_History.png",
    'scatter_plot': r"../results/rfkq\DNN_Stable_Scatter.png",
    'model_file': r"../results/rfkq\DNN_Stable_model.keras",
}

TIFF_PATHS = {
    'dsm_bathy':   r"../data/raster\DEM_ThuaThienHue.tif",
    'slope':       r"../data/raster\Slope_ThuaThienHue.tif",
    'aspect':      r"../data/raster\Aspect_ThuaThienHue.tif",
    'curvature':   r"../data/raster\Curvature_ThuaThienHue.tif",
    'hillshade':   r"../data/raster\Hillshade_ThuaThienHue.tif",
    'roughness':   r"../data/raster\Roughness_ThuaThienHue.tif",
    'spi':         r"../data/raster\spi_10m.tif",
    'tpi':         r"../data/raster\TPI_ThuaThienHue.tif",
    'twi':         r"../data/raster\twi_10m.tif",
    'flow_accum':  r"../data/raster\flow_accum_10m.tif",
    'hand30_100':  r"../data/raster\HAND_30_100_ThuaThienHue.tif",
    'distance_r':  r"../data/raster\distance_r final.tif",
    'density_ri':  r"../data/raster\density_r.tif",
    'gsw_occure':  r"../data/raster\gsw.tif",
    'precip_s_1':  r"../data/raster\Hue_Rainfall_Total_1999_10mT12.tif",
    'VLUCD_L2_3':  r"../data/raster\lulc hue.tif",
}

TARGET_COL = 'Flood_1999'
NODATA_VAL = -9999
TARGET_RESOLUTION = 30


def build_simple_dnn(input_dim):
    """Model đơn giản, ổn định"""
    inputs = layers.Input(shape=(input_dim,))
    
    # Simple architecture
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Output
    outputs = layers.Dense(1, activation='relu', kernel_initializer='glorot_uniform')(x)  # ReLU để đảm bảo > 0
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='SimpleDNN')
    return model




def main():
    print("🌊 DNN ỔN ĐỊNH - DỰ BÁO NGẬP LỤT\n")
    
    # 1. Load data
    print("1️⃣  Load dữ liệu...")
    df = pd.read_csv(CONFIG['csv_file'])
    valid_features = {k: v for k, v in TIFF_PATHS.items() if v and os.path.exists(v)}
    feature_names = list(valid_features.keys())
    
    for feat, path in tqdm(valid_features.items(), desc="   Sampling"):
        df[feat] = extract_values_at_points(path, df['lon_new'], df['lat_new'])
    
    df_clean = df.dropna(subset=[TARGET_COL])
    print(f"   -> {len(df_clean)} samples, {len(feature_names)} features")
    
    # 2. Prepare - KHÔNG TẠO INTERACTION FEATURES (giữ đơn giản)
    print("\n2️⃣  Chuẩn bị dữ liệu...")
    X = df_clean[feature_names].values.astype(np.float32)
    y = df_clean[TARGET_COL].values.astype(np.float32)
    
    # KIỂM TRA DỮ LIỆU
    print(f"   X shape: {X.shape}")
    print(f"   y range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Loại bỏ outliers
    mask = (y >= 0) & (y <= 5)  # Chỉ giữ 0-5m
    X = X[mask]
    y = y[mask]
    print(f"   Sau khi lọc outliers: {len(y)} samples")
    
    # Replace inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # RobustScaler (tốt hơn StandardScaler cho data có outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clip extreme values sau khi scale
    X_scaled = np.clip(X_scaled, -10, 10)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 3. Build & Train
    print("\n3️⃣  Training...")
    model = build_simple_dnn(input_dim=X_train.shape[1])
    
    # Optimizer với gradient clipping
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='huber',  # Huber loss robust hơn MSE
        metrics=['mae', 'mse']
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=1e-6,
        verbose=1
    )
    
    # Terminate on NaN
    terminate_nan = callbacks.TerminateOnNaN()
    
    print("\n🚀 Training...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=300,
        batch_size=16,  # Batch nhỏ
        callbacks=[early_stop, reduce_lr, terminate_nan],
        verbose=1
    )
    
    print(f"\n✅ Done! ({len(history.history['loss'])} epochs)")
    
    # 4. Evaluate - GIỐNG RF/XGB
    print("\n4️⃣  Đánh giá model...")
    y_pred = model.predict(X_test, batch_size=128, verbose=0).flatten()
    
    # Clip predictions
    y_pred = np.clip(y_pred, 0, 5)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Tính MAPE
    mask_nonzero = y_test != 0
    mape = np.mean(np.abs((y_test[mask_nonzero] - y_pred[mask_nonzero]) / y_test[mask_nonzero])) * 100 if np.any(mask_nonzero) else 0
    
    print(f"\n📊 ĐÁNH GIÁ MÔ HÌNH DNN:")
    print(f"   • RMSE: {rmse:.4f} m")
    print(f"   • MAE: {mae:.4f} m")
    print(f"   • R²: {r2:.4f}")
    print(f"   • MAPE: {mape:.2f}%")
    print(f"\n   🔍 Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"   🔍 Prediction std: {y_pred.std():.4f}")
    
    if y_pred.std() < 0.01:
        print("   ⚠️  WARNING: Predictions có variance rất thấp!")
    
    # Save comprehensive report - GIỐNG RF/XGB
    print(f"\n   -> Đang lưu báo cáo đầy đủ...")
    with open(CONFIG['evaluation_report'], 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH DNN - DỰ BÁO NGẬP LỤT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. THÔNG SỐ MÔ HÌNH DNN:\n")
        f.write(f"   - Device: {'GPU' if gpus else 'CPU'}\n")
        if gpus:
            f.write(f"   - GPU Info: {gpus[0].name}\n")
        f.write(f"   - Architecture: Simple DNN (128-64-32)\n")
        f.write(f"   - Activation: ReLU\n")
        f.write(f"   - Output Activation: ReLU (non-negative)\n")
        f.write(f"   - Regularization: Dropout (0.1) + Batch Normalization\n")
        f.write(f"   - Optimizer: Adam (LR=0.001, clipnorm=1.0)\n")
        f.write(f"   - Loss Function: Huber Loss\n")
        f.write(f"   - Batch Size: 16\n")
        f.write(f"   - Số epochs trained: {len(history.history['loss'])}\n")
        f.write(f"   - Best epoch: {len(history.history['loss']) - 50}\n")
        f.write(f"   - Số lượng features: {len(feature_names)}\n")
        f.write(f"   - Tổng số điểm dữ liệu: {len(X_train) + len(X_test)}\n")
        f.write(f"   - Số điểm huấn luyện (80%): {len(X_train)}\n")
        f.write(f"   - Số điểm kiểm tra (20%): {len(X_test)}\n\n")
        
        f.write("2. CHỈ SỐ ĐÁNH GIÁ (trên tập test):\n")
        f.write(f"   - RMSE: {rmse:.4f} m\n")
        f.write(f"   - MAE: {mae:.4f} m\n")
        f.write(f"   - R²: {r2:.4f}\n")
        f.write(f"   - MAPE: {mape:.2f}%\n\n")
        
        f.write("3. THỐNG KÊ DỰ BÁO:\n")
        f.write(f"   - Giá trị nhỏ nhất: {y_pred.min():.4f} m\n")
        f.write(f"   - Giá trị lớn nhất: {y_pred.max():.4f} m\n")
        f.write(f"   - Giá trị trung bình: {y_pred.mean():.4f} m\n")
        f.write(f"   - Độ lệch chuẩn: {y_pred.std():.4f} m\n\n")
        
        f.write("4. TRAINING HISTORY:\n")
        f.write(f"   - Initial Training Loss: {history.history['loss'][0]:.4f}\n")
        f.write(f"   - Final Training Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"   - Initial Validation Loss: {history.history['val_loss'][0]:.4f}\n")
        f.write(f"   - Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
        f.write(f"   - Best Validation Loss: {min(history.history['val_loss']):.4f}\n")
        f.write(f"   - Initial MAE: {history.history['mae'][0]:.4f} m\n")
        f.write(f"   - Final MAE: {history.history['mae'][-1]:.4f} m\n\n")
        
        f.write("5. FEATURES SỬ DỤNG:\n")
        for i, feat in enumerate(feature_names, 1):
            f.write(f"   {i:2d}. {feat}\n")
        
        f.write("\n6. GHI CHÚ:\n")
        f.write("   - Model sử dụng RobustScaler để chuẩn hóa dữ liệu\n")
        f.write("   - Huber Loss giúp robust với outliers\n")
        f.write("   - Gradient clipping tránh exploding gradients\n")
        f.write("   - Early stopping với patience=50\n")
        f.write("   - Learning rate reduction tự động khi plateau\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"   ✅ Đã lưu báo cáo: {CONFIG['evaluation_report']}")
    
    # Plots - GIỐNG RF/XGB
    print(f"   -> Đang tạo biểu đồ...")
    
    # Plot 1: Training History
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss (Huber)', fontsize=11)
    plt.title('Training & Validation Loss', fontsize=12, weight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('MAE (m)', fontsize=11)
    plt.title('Training & Validation MAE', fontsize=12, weight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CONFIG['training_history_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu Training History: {CONFIG['training_history_plot']}")
    
    # Plot 2: Predicted vs Actual - GIỐNG XGB
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, s=20, c='blue', edgecolors='navy', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Flood Depth (m)', fontsize=12)
    plt.ylabel('Predicted Flood Depth (m)', fontsize=12)
    plt.title(f'Predicted vs Actual (R² = {r2:.4f}) - DNN Model', fontsize=13, weight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text box with metrics
    textstr = f'RMSE = {rmse:.4f} m\nMAE = {mae:.4f} m\nMAPE = {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(CONFIG['scatter_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu Scatter Plot: {CONFIG['scatter_plot']}")
    
    # Plot 3: Residuals Distribution
    residuals_plot = CONFIG['scatter_plot'].replace('Scatter', 'Residuals')
    plt.figure(figsize=(12, 4))
    
    residuals = y_test - y_pred
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Flood Depth (m)', fontsize=11)
    plt.ylabel('Residuals (m)', fontsize=11)
    plt.title('Residual Plot', fontsize=12, weight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals (m)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Residual Distribution', fontsize=12, weight='bold')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(residuals_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu Residuals Plot: {residuals_plot}")
    
    # Save model
    model.save(CONFIG['model_file'])
    print(f"   ✅ Đã lưu model: {CONFIG['model_file']}")
    
    # 5. Predict raster
    print("\n5️⃣  Dự báo toàn bộ raster...")
    
    ref_path = TIFF_PATHS['dsm_bathy']
    with rasterio.open(ref_path) as src_ref:
        degrees_per_30m = TARGET_RESOLUTION / 111000.0
        scale = degrees_per_30m / src_ref.res[0]
        new_width = int(src_ref.width / scale)
        new_height = int(src_ref.height / scale)
        
        print(f"   -> Output size: {new_width} x {new_height} pixels")
        print(f"   -> Total pixels: {new_width * new_height:,} (~{new_width*new_height/1e6:.2f}M)")
        
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
    
    src_files = {feat: rasterio.open(path) for feat, path in valid_features.items()}
    temp_output = CONFIG['output_file'].replace('.tif', '_temp.tif')
    
    try:
        with rasterio.open(temp_output, 'w', **out_meta) as dst:
            for row_start in tqdm(range(0, new_height, 512), desc="   Đang dự báo"):
                row_end = min(row_start + 512, new_height)
                rows = row_end - row_start
                cols = new_width
                
                window_10m = Window(0, int(row_start*scale), int(cols*scale), int(rows*scale))
                
                X_block = np.zeros((rows*cols, len(valid_features)), dtype=np.float32)
                mask_valid = np.ones((rows*cols), dtype=bool)
                
                for i, feat in enumerate(valid_features.keys()):
                    try:
                        data = src_files[feat].read(
                            1, window=window_10m, out_shape=(rows, cols),
                            resampling=Resampling.bilinear, boundless=True, fill_value=0
                        ).flatten()
                        X_block[:, i] = data
                        if feat == 'dsm_bathy':
                            mask_valid = (data > -100) & (data != (src_files[feat].nodata or -9999))
                    except:
                        X_block[:, i] = 0
                
                y_block = np.full(rows*cols, NODATA_VAL, dtype=np.float32)
                if np.any(mask_valid):
                    X_valid = np.nan_to_num(X_block[mask_valid], nan=0.0, posinf=0.0, neginf=0.0)
                    X_valid_scaled = scaler.transform(X_valid)
                    X_valid_scaled = np.clip(X_valid_scaled, -10, 10)
                    preds = model.predict(X_valid_scaled, batch_size=4096, verbose=0).flatten()
                    y_block[mask_valid] = np.clip(preds, 0, 5)
                
                dst.write(y_block.reshape(rows, cols), 1, window=Window(0, row_start, cols, rows))
    
    finally:
        for src in src_files.values():
            src.close()
    
    print("   ✅ Đã hoàn tất dự báo!")
    
    # 6. Clip
    if CONFIG['shapefile'] and os.path.exists(CONFIG['shapefile']):
        clip_raster_by_shapefile(temp_output, CONFIG['shapefile'], CONFIG['output_file'])
        if os.path.exists(temp_output):
            os.remove(temp_output)
    else:
        os.rename(temp_output, CONFIG['output_file'])
    
    # 7. Export CSV với toàn bộ dữ liệu dự báo
    print("\n7️⃣  Đang xuất CSV với toàn bộ điểm dự báo...")
    
    try:
        # Đọc lại file output để export CSV
        with rasterio.open(CONFIG['output_file']) as src:
            depth_data = src.read(1)
            transform = src.transform
            height, width = depth_data.shape
            
            # Tạo lưới tọa độ
            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            
            # Mask valid data
            valid_mask = depth_data != NODATA_VAL
            
            if np.any(valid_mask):
                # Lấy tọa độ của pixels hợp lệ
                valid_rows = rows[valid_mask]
                valid_cols = cols[valid_mask]
                valid_depths = depth_data[valid_mask]
                
                # Chuyển đổi pixel coords sang geo coords
                xs, ys = rasterio.transform.xy(transform, valid_rows, valid_cols)
                
                # Tạo DataFrame
                csv_data = pd.DataFrame({
                    'lon': xs,
                    'lat': ys,
                    'predicted_depth_m': valid_depths,
                    'flood_class': classify_flood_depth(valid_depths)
                })
                
                # Làm tròn để giảm kích thước file
                csv_data['lon'] = csv_data['lon'].round(6)
                csv_data['lat'] = csv_data['lat'].round(6)
                csv_data['predicted_depth_m'] = csv_data['predicted_depth_m'].round(4)
                
                # Xóa file CSV cũ nếu tồn tại
                if os.path.exists(CONFIG['output_csv']):
                    try:
                        os.remove(CONFIG['output_csv'])
                    except:
                        pass
                
                # Lưu CSV
                csv_data.to_csv(CONFIG['output_csv'], index=False)
                
                print(f"   ✅ Đã xuất {len(csv_data):,} điểm vào CSV")
                print(f"   📁 {CONFIG['output_csv']}")
                
                # Thống kê ngắn gọn
                print(f"\n   📊 Thống kê CSV:")
                print(f"      - Tổng điểm: {len(csv_data):,}")
                print(f"      - Độ sâu TB: {csv_data['predicted_depth_m'].mean():.4f} m")
                print(f"      - Độ sâu Max: {csv_data['predicted_depth_m'].max():.4f} m")
                print(f"      - Kích thước file: {os.path.getsize(CONFIG['output_csv']) / 1024 / 1024:.2f} MB")
            else:
                print("   ⚠️  Không có dữ liệu hợp lệ để xuất CSV")
                
    except Exception as e:
        print(f"   ❌ Lỗi khi xuất CSV: {e}")
        print(f"   ⚠️  Tiếp tục mà không có file CSV...")
    
    # 8. Classification
    print("\n8️⃣  Tạo bản đồ phân loại...")
    create_flood_classification_map(CONFIG['output_file'], CONFIG['output_classified'], CONFIG['output_legend'])
    
    print(f"\n🎉 HOÀN TẤT - DNN MODEL!")
    print("=" * 70)
    print("📁 CÁC FILE OUTPUT:")
    print(f"   1. Raster độ sâu 30m: {CONFIG['output_file']}")
    print(f"   2. Raster phân loại 6 cấp: {CONFIG['output_classified']}")
    print(f"   3. Chú giải bản đồ: {CONFIG['output_legend']}")
    print(f"   4. CSV toàn bộ điểm: {CONFIG['output_csv']}")
    print(f"   5. Model file: {CONFIG['model_file']}")
    print(f"   6. Báo cáo đánh giá: {CONFIG['evaluation_report']}")
    print(f"   7. Training History plot: {CONFIG['training_history_plot']}")
    print(f"   8. Scatter plot: {CONFIG['scatter_plot']}")
    print(f"   9. Residuals plot: {residuals_plot}")
    print("=" * 70)

if __name__ == "__main__":
    main()
