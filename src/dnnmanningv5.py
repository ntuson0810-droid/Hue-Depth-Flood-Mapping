"""
================================================================================
DỰ BÁO ĐỘ SÂU NGẬP LỤT - DNN v6 (Bounded Output)
================================================================================
Cải tiến v5 → v6:
  ✅ Output activation: ReLU → Sigmoid × TARGET_MAX (bounded [0, 5m])
  ✅ Target normalization: y → y / TARGET_MAX (training trên [0, 1])
  ✅ Loại bỏ hoàn toàn spike artifact tại 5m
  ✅ Final clip 5.0 → 3.5m (phù hợp dataset thực tế, dataset max ~3m)
  ✅ Tăng dropout 0.2 → 0.3 chống extrapolation
  ✅ Giữ nguyên toàn bộ kỹ thuật: Manning ESA, Cap-only, Bathtub, smoothing
================================================================================
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
from rasterio.transform import Affine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
import os
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam

# Cấu hình GPU
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

# ==============================================================================
# 👇 KHU VỰC CẤU HÌNH 👇
# ==============================================================================
CONFIG = {
    'csv_file': r"../data/raw/FloodMarks1999.csv",
    'shapefile': r"../data/spatial/Hue shapefile\hue.shp",

    'output_file_raw': r"../results/rfkq\DNNv6_KetQua_Hue_30m_RAW.tif",
    'output_file': r"../results/rfkq\DNNv6_KetQua_Hue_30m.tif",

    'output_classified': r"../results/rfkq\DNNv6_KetQua_Hue_30m_PhanLoai.tif",
    'output_legend': r"../results/rfkq\DNNv6_Flood_Classification_Legend.png",
    'output_csv': r"../results/rfkq\DNNv6_KetQua_Hue_30m_FullData.csv",
    'evaluation_report': r"../results/rfkq\DNNv6_Model_Evaluation_Report.txt",
    'training_history_plot': r"../results/rfkq\DNNv6_Training_History.png",
    'scatter_plot': r"../results/rfkq\DNNv6_Predicted_vs_Actual.png",
    'residuals_plot': r"../results/rfkq\DNNv6_Residuals.png",
    'permutation_plot': r"../results/rfkq\DNNv6_Permutation_Importance.png",
    'model_file': r"../results/rfkq\DNNv6_model.keras",
    'scaler_file': r"../results/rfkq\DNNv6_scaler.joblib",

    'manning_raster': r"../data/raster\Manning_N_ThuaThienHue_ESA.tif",
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
TARGET_MAX = 5.0  # ⭐ Giá trị max để normalize y → sigmoid output bounded

# ==============================================================================
# 👇 BẢNG TRA MANNING'S N CHO ESA WORLDCOVER 👇
# ==============================================================================
ESA_MANNING_LOOKUP = {
    10:  0.120, 20:  0.060, 30:  0.035, 40:  0.040, 50:  0.020,
    60:  0.025, 70:  0.012, 80:  0.030, 90:  0.060, 95:  0.100, 100: 0.025,
}
DEFAULT_MANNING = 0.040
CATEGORICAL_FEATURES = {'VLUCD_L2_3'}


# ==============================================================================
# HÀM: TẠO RASTER MANNING'S N
# ==============================================================================




# ==============================================================================
# HÀM: INTERACTION FEATURES (DÙNG CẢ TRAIN & PREDICT)
# ==============================================================================


# ==============================================================================
# HÀM: HẬU XỬ LÝ THỦY LỰC (CAP-ONLY)
# ==============================================================================


# ==============================================================================
# HÀM: BATHTUB SPREADING
# ==============================================================================


# ==============================================================================
# HÀM HỖ TRỢ
# ==============================================================================


# ==============================================================================
# ⭐ HÀM v6: BUILD DNN VỚI SIGMOID OUTPUT (BOUNDED [0, TARGET_MAX])
# ==============================================================================
def build_dnn(input_dim, target_max=TARGET_MAX):
    """
    DNN v6 với SIGMOID output × TARGET_MAX:
      - Output nằm trong khoảng [0, target_max] một cách TỰ NHIÊN
      - Loại bỏ extrapolation problem của ReLU
      - Không tạo spike artifact ở giá trị clip
    """
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)  # ⭐ tăng 0.2 → 0.3

    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    # ⭐ V6: SIGMOID OUTPUT - bounded [0, 1] tự nhiên
    # Sẽ multiply với target_max sau predict để có [0, target_max]
    outputs = layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform',
                           name='depth_normalized')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='DNN_v6')
    return model








# ==============================================================================
# HÀM: ĐÁNH GIÁ NÂNG CAO
# ==============================================================================
def evaluate_model_advanced(y_true, y_pred, history, feature_names, gpus_info,
                              n_train, n_test, model_type="DNN"):
    print(f"\n📊 ĐÁNH GIÁ MÔ HÌNH {model_type}:")
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)
    mask = y_true_np != 0
    mape = np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100 if np.any(mask) else 0

    print(f"\n   ── METRIC TỔNG (n={len(y_true_np)}) ──")
    print(f"   • R²: {r2:.4f} | RMSE: {rmse:.4f}m | MAE: {mae:.4f}m | MAPE: {mape:.2f}%")

    flood_mask = y_true_np > 0
    n_flood = flood_mask.sum()
    if n_flood >= 5:
        rmse_flood = np.sqrt(mean_squared_error(y_true_np[flood_mask], y_pred_np[flood_mask]))
        mae_flood = mean_absolute_error(y_true_np[flood_mask], y_pred_np[flood_mask])
        if n_flood >= 10 and np.var(y_true_np[flood_mask]) > 0:
            r2_flood = r2_score(y_true_np[flood_mask], y_pred_np[flood_mask])
        else:
            r2_flood = float('nan')
        bias_flood = (y_pred_np[flood_mask] - y_true_np[flood_mask]).mean()
        print(f"\n   ── METRIC CHỈ VÙNG NGẬP (Actual > 0, n={n_flood}) ──")
        print(f"   • R² flood-only:   {r2_flood:.4f}")
        print(f"   • RMSE flood-only: {rmse_flood:.4f}m | MAE: {mae_flood:.4f}m | Bias: {bias_flood:+.4f}m")
    else:
        r2_flood = rmse_flood = mae_flood = bias_flood = float('nan')

    print(f"\n   ── RMSE THEO CẤP ĐỘ NGẬP ──")
    bins = [(0, 0, 'Không ngập'), (0.001, 0.5, '0-0.5m'),
            (0.5, 1.0, '0.5-1m'), (1.0, 1.5, '1-1.5m'),
            (1.5, 2.0, '1.5-2m'), (2.0, 100, '>2m')]
    bin_results = []
    for lo, hi, label in bins:
        if lo == 0 and hi == 0:
            m = y_true_np == 0
        else:
            m = (y_true_np > lo) & (y_true_np <= hi)
        if m.sum() > 0:
            rmse_b = np.sqrt(mean_squared_error(y_true_np[m], y_pred_np[m]))
            bias_b = (y_pred_np[m] - y_true_np[m]).mean()
            print(f"   • {label:<12s} n={m.sum():>4} | RMSE={rmse_b:.3f}m | Bias={bias_b:+.3f}m")
            bin_results.append((label, int(m.sum()), float(rmse_b), float(bias_b)))

    try:
        from sklearn.metrics import accuracy_score
        y_true_cls = classify_flood_depth(y_true_np)
        y_pred_cls = classify_flood_depth(np.maximum(y_pred_np, 0))
        acc = accuracy_score(y_true_cls, y_pred_cls)
        within_1 = (np.abs(y_true_cls.astype(int) - y_pred_cls.astype(int)) <= 1).mean()
        print(f"\n   ── CLASSIFICATION METRICS ──")
        print(f"   • Accuracy: {acc:.4f} | Accuracy ±1 cấp: {within_1:.4f}")
    except:
        acc = within_1 = float('nan')

    report_path = CONFIG['evaluation_report']
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"BÁO CÁO ĐÁNH GIÁ MÔ HÌNH {model_type} v6 (Bounded Output)\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. THÔNG SỐ MÔ HÌNH:\n")
        f.write(f"   - Device: {'GPU' if gpus_info else 'CPU'}\n")
        f.write(f"   - Architecture: DNN (256-128-64-32) + L2 + Dropout 0.2-0.3\n")
        f.write(f"   - Output Activation: ⭐ SIGMOID × {TARGET_MAX} (bounded [0, {TARGET_MAX}]m)\n")
        f.write(f"   - Target normalization: y / {TARGET_MAX} → train trên [0, 1]\n")
        f.write(f"   - Optimizer: Adam (LR=0.001, clipnorm=1.0)\n")
        f.write(f"   - Loss: Huber (trên y normalized)\n")
        f.write(f"   - Scaler: RobustScaler\n")
        f.write(f"   - Epochs trained: {len(history.history['loss'])}\n")
        f.write(f"   - Features: {len(feature_names)} | Train/Test: {n_train}/{n_test}\n")
        f.write(f"   - Manning lookup: ESA WorldCover\n")
        f.write(f"   - Hậu xử lý: CAP-ONLY + Bathtub + Final Clip 3.5m\n\n")
        f.write("2. CHỈ SỐ TỔNG:\n")
        f.write(f"   - R²: {r2:.4f} | RMSE: {rmse:.4f}m | MAE: {mae:.4f}m | MAPE: {mape:.2f}%\n\n")
        f.write(f"3. CHỈ SỐ CHỈ VÙNG NGẬP (n={n_flood}):\n")
        f.write(f"   - R² flood-only: {r2_flood:.4f}\n")
        f.write(f"   - RMSE flood-only: {rmse_flood:.4f}m\n")
        f.write(f"   - Bias: {bias_flood:+.4f}m\n\n")
        f.write("4. RMSE THEO CẤP ĐỘ:\n")
        for label, n, rmse_b, bias_b in bin_results:
            f.write(f"   - {label:<12s} n={n:>4} | RMSE={rmse_b:.3f}m | Bias={bias_b:+.3f}m\n")
        f.write("\n5. CLASSIFICATION METRICS:\n")
        f.write(f"   - Accuracy (exact): {acc:.4f}\n")
        f.write(f"   - Accuracy (±1 cấp): {within_1:.4f}\n\n")
        f.write("6. TRAINING HISTORY:\n")
        f.write(f"   - Initial Loss: {history.history['loss'][0]:.4f}\n")
        f.write(f"   - Final Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"   - Best Val Loss: {min(history.history['val_loss']):.4f}\n\n")
        f.write(f"7. FEATURES ({len(feature_names)}):\n")
        for i, feat in enumerate(feature_names, 1):
            f.write(f"   {i:2d}. {feat}\n")
        f.write("\n" + "=" * 70 + "\n")
    print(f"\n   ✅ Báo cáo: {report_path}")

    # Scatter
    plt.figure(figsize=(9, 8))
    dry_m = y_true_np == 0
    plt.scatter(y_true_np[dry_m], y_pred_np[dry_m], alpha=0.4, s=12,
                color='gray', label=f'Không ngập (n={dry_m.sum()})')
    plt.scatter(y_true_np[~dry_m], y_pred_np[~dry_m], alpha=0.6, s=20,
                color='steelblue', edgecolor='black', linewidth=0.3,
                label=f'Có ngập (n={(~dry_m).sum()})')
    max_val = max(y_true_np.max(), y_pred_np.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='y = x')
    plt.xlabel('Actual Flood Depth (m)', fontsize=12)
    plt.ylabel('Predicted Flood Depth (m)', fontsize=12)
    plt.title(f'Predicted vs Actual - {model_type} (Sigmoid Output)\n'
              f'R² overall = {r2:.4f} | R² flood-only = {r2_flood:.4f}',
              fontsize=12, weight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['scatter_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Scatter: {CONFIG['scatter_plot']}")
    return rmse, mae, r2, mape, r2_flood, rmse_flood


# ==============================================================================
# HÀM: PERMUTATION IMPORTANCE
# ==============================================================================
def compute_permutation_importance_dnn(model, X_test, y_test, feature_names,
                                         n_repeats=5, batch_size=128):
    print(f"\n🔄 TÍNH PERMUTATION IMPORTANCE ({n_repeats} repeats)...")
    y_pred_baseline = model.predict(X_test, batch_size=batch_size, verbose=0).flatten() * TARGET_MAX
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    print(f"   Baseline R²: {baseline_r2:.4f}")

    importances_mean = np.zeros(len(feature_names))
    importances_std = np.zeros(len(feature_names))
    rng = np.random.RandomState(42)

    for i, feat in enumerate(tqdm(feature_names, desc="   Permuting")):
        scores = []
        for r in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            y_pred_perm = model.predict(X_perm, batch_size=batch_size, verbose=0).flatten() * TARGET_MAX
            r2_perm = r2_score(y_test, y_pred_perm)
            scores.append(baseline_r2 - r2_perm)
        importances_mean[i] = np.mean(scores)
        importances_std[i] = np.std(scores)

    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'PermImp_Mean': importances_mean,
        'PermImp_Std': importances_std,
    }).sort_values('PermImp_Mean', ascending=False)

    print(f"\n   ── TOP 15 PERMUTATION IMPORTANCE ──")
    for idx, row in perm_df.head(15).iterrows():
        print(f"   • {row['Feature']:<25s} {row['PermImp_Mean']:+.4f} ± {row['PermImp_Std']:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    top = perm_df.sort_values('PermImp_Mean', ascending=True).tail(20)
    ax.barh(top['Feature'], top['PermImp_Mean'], xerr=top['PermImp_Std'],
            color='steelblue', edgecolor='black', error_kw={'ecolor': 'red'})
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Permutation Importance (drop in R²)')
    ax.set_title('Top 20 Permutation Importance - DNN v6')
    plt.tight_layout()
    plt.savefig(CONFIG['permutation_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {CONFIG['permutation_plot']}")
    return perm_df


# ==============================================================================
# HÀM CHÍNH
# ==============================================================================
def main():
    print("🌊 DNN v6 - SIGMOID BOUNDED OUTPUT + Manning + Cap-Only + Bathtub")
    print("=" * 78)

    output_dir = os.path.dirname(CONFIG['output_file'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # BƯỚC 0: MANNING
    print("\n0️⃣  KHỞI TẠO MANNING'S N RASTER...")
    if os.path.exists(TIFF_PATHS['VLUCD_L2_3']):
        manning_dir = os.path.dirname(CONFIG['manning_raster'])
        if manning_dir and not os.path.exists(manning_dir):
            os.makedirs(manning_dir)
        create_manning_raster_from_lulc(TIFF_PATHS['VLUCD_L2_3'], CONFIG['manning_raster'])
        TIFF_PATHS['manning_n'] = CONFIG['manning_raster']

    # BƯỚC 1: LOAD
    print("\n1️⃣  LOAD DỮ LIỆU...")
    df = pd.read_csv(CONFIG['csv_file'])
    valid_features = {k: v for k, v in TIFF_PATHS.items() if v and os.path.exists(v)}
    feature_names = list(valid_features.keys())
    print(f"   -> {len(valid_features)} file Tiff hợp lệ")
    for feat, path in tqdm(valid_features.items(), desc="   Sampling"):
        df[feat] = extract_values_at_points(path, df['lon_new'], df['lat_new'])
    df_clean = df.dropna(subset=[TARGET_COL]).copy()
    print(f"   -> {len(df_clean)} samples")

    # BƯỚC 2: FEATURE ENGINEERING
    print("\n2️⃣  TẠO INTERACTION FEATURES...")
    data_dict_train = {feat: df_clean[feat].values.astype(np.float32) for feat in feature_names}
    interaction_arrs, interaction_names = create_interaction_features(data_dict_train)
    for arr, name in zip(interaction_arrs, interaction_names):
        df_clean[name] = arr
        feature_names.append(name)
    print(f"   -> Tổng features: {len(feature_names)}")

    # BƯỚC 3: CHUẨN BỊ
    print("\n3️⃣  CHUẨN BỊ DỮ LIỆU...")
    X = df_clean[feature_names].values.astype(np.float32)
    y = df_clean[TARGET_COL].values.astype(np.float32)
    print(f"   X shape: {X.shape} | y range: [{y.min():.4f}, {y.max():.4f}]")

    # Loại outliers
    mask = (y >= 0) & (y <= TARGET_MAX)
    X = X[mask]
    y = y[mask]
    print(f"   Sau lọc outliers (0-{TARGET_MAX}m): {len(y)} samples")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ⭐ V6: NORMALIZE y → [0, 1] cho sigmoid output
    y_normalized = y / TARGET_MAX
    print(f"   ⭐ Y NORMALIZED: y / {TARGET_MAX} → range [{y_normalized.min():.4f}, {y_normalized.max():.4f}]")

    # Scaler cho features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -10, 10)

    # Split (chia trên y normalized)
    X_train, X_test, y_train_norm, y_test_norm = train_test_split(
        X_scaled, y_normalized, test_size=0.2, random_state=42, shuffle=True
    )
    # Lưu y_test ở scale gốc cho đánh giá
    y_test_original = y_test_norm * TARGET_MAX
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # BƯỚC 4: BUILD & TRAIN
    print("\n4️⃣  BUILD & TRAIN DNN v6 (SIGMOID OUTPUT)...")
    model = build_dnn(input_dim=X_train.shape[1], target_max=TARGET_MAX)

    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='huber',
        metrics=['mae', 'mse']
    )
    print("\n📋 Model Summary:")
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=50, restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1
    )
    terminate_nan = callbacks.TerminateOnNaN()

    print("\n🚀 Training (trên y NORMALIZED)...\n")
    history = model.fit(
        X_train, y_train_norm,
        validation_data=(X_test, y_test_norm),
        epochs=300, batch_size=16,
        callbacks=[early_stop, reduce_lr, terminate_nan],
        verbose=1
    )
    print(f"\n✅ Training xong! ({len(history.history['loss'])} epochs)")

    # BƯỚC 5: ĐÁNH GIÁ (trên scale gốc)
    y_pred_norm = model.predict(X_test, batch_size=128, verbose=0).flatten()
    y_pred = y_pred_norm * TARGET_MAX  # ⭐ Chuyển về scale gốc

    print(f"\n   🔍 Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"   🔍 Prediction std: {y_pred.std():.4f}")

    evaluate_model_advanced(y_test_original, y_pred, history, feature_names, gpus,
                              len(X_train), len(X_test), model_type="DNN v6")

    # PERMUTATION IMPORTANCE
    try:
        perm_df = compute_permutation_importance_dnn(
            model, X_test, y_test_original, feature_names, n_repeats=5
        )
    except Exception as e:
        print(f"   ⚠️  Lỗi permutation: {e}")
        perm_df = None

    # LƯU MODEL & SCALER
    print(f"\n   -> Lưu model...")
    model.save(CONFIG['model_file'])
    print(f"   ✅ Model: {CONFIG['model_file']}")

    import joblib
    joblib.dump({
        'scaler': scaler,
        'feature_names': feature_names,
        'target_max': TARGET_MAX,
        'esa_manning_lookup': ESA_MANNING_LOOKUP,
    }, CONFIG['scaler_file'])
    print(f"   ✅ Scaler: {CONFIG['scaler_file']}")

    # PLOTS
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss (Huber on normalized y)')
    plt.title('Training & Validation Loss', weight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('MAE (normalized scale)')
    plt.title('Training & Validation MAE', weight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['training_history_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Training History: {CONFIG['training_history_plot']}")

    # Residuals
    plt.figure(figsize=(12, 4))
    residuals = y_test_original - y_pred
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted (m)'); plt.ylabel('Residuals (m)')
    plt.title('Residual Plot', weight='bold')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals (m)'); plt.ylabel('Frequency')
    plt.title('Residual Distribution', weight='bold')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['residuals_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Residuals: {CONFIG['residuals_plot']}")

    # BƯỚC 6: DỰ BÁO RA RASTER 30M
    print("\n5️⃣  DỰ BÁO RA RASTER 30M...")
    ref_path = TIFF_PATHS.get('dsm_bathy')
    if not ref_path or not os.path.exists(ref_path):
        print("❌ Không có dsm_bathy!")
        return

    with rasterio.open(ref_path) as src_ref:
        print(f"   -> Gốc: {src_ref.width} x {src_ref.height}")
        degrees_per_30m = TARGET_RESOLUTION / 111000.0
        scale = degrees_per_30m / src_ref.res[0]
        new_width = int(src_ref.width / scale)
        new_height = int(src_ref.height / scale)
        print(f"   -> Output: {new_width} x {new_height}")

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
    temp_output = CONFIG['output_file_raw'].replace('.tif', '_temp.tif')
    base_features = list(valid_features.keys())
    block_size = 512

    try:
        with rasterio.open(temp_output, 'w', **out_meta) as dst:
            for row_start in tqdm(range(0, new_height, block_size), desc="   Predict"):
                row_end = min(row_start + block_size, new_height)
                rows = row_end - row_start
                cols = new_width
                window_30m = Window(0, row_start, cols, rows)
                window_10m = Window(0, int(row_start * scale), int(cols * scale), int(rows * scale))

                feature_data = {}
                mask_valid = np.ones((rows * cols), dtype=bool)
                for feat in base_features:
                    try:
                        resamp = Resampling.nearest if feat in CATEGORICAL_FEATURES else Resampling.bilinear
                        data = src_files[feat].read(
                            1, window=window_10m, out_shape=(rows, cols),
                            resampling=resamp, boundless=True, fill_value=0
                        ).flatten().astype(np.float32)
                        feature_data[feat] = data
                        if feat == 'dsm_bathy':
                            nodata = src_files[feat].nodata if src_files[feat].nodata is not None else -9999
                            mask_valid = (data > -100) & (data != nodata)
                    except Exception as e:
                        print(f"\n   ⚠️  Lỗi {feat}: {e}")
                        feature_data[feat] = np.zeros(rows * cols, dtype=np.float32)

                interaction_arrs_pred, interaction_names_pred = create_interaction_features(feature_data)

                X_block_full = np.zeros((rows * cols, len(feature_names)), dtype=np.float32)
                for i, fname in enumerate(feature_names):
                    if fname in feature_data:
                        X_block_full[:, i] = feature_data[fname]
                    elif fname in interaction_names_pred:
                        idx = interaction_names_pred.index(fname)
                        X_block_full[:, i] = interaction_arrs_pred[idx]

                y_block = np.full(rows * cols, NODATA_VAL, dtype=np.float32)
                if np.any(mask_valid):
                    X_valid = np.nan_to_num(X_block_full[mask_valid], nan=0.0, posinf=0.0, neginf=0.0)
                    X_valid_scaled = scaler.transform(X_valid)
                    X_valid_scaled = np.clip(X_valid_scaled, -10, 10)
                    preds_norm = model.predict(X_valid_scaled, batch_size=4096, verbose=0).flatten()
                    preds = preds_norm * TARGET_MAX  # ⭐ Chuyển về scale gốc
                    # Sigmoid đã bounded [0,1] → preds ∈ [0, TARGET_MAX] tự nhiên, không cần clip
                    y_block[mask_valid] = preds

                dst.write(y_block.reshape(rows, cols), 1, window=window_30m)
    finally:
        for src in src_files.values():
            src.close()

    # BƯỚC 7: CẮT THEO SHAPEFILE
    if CONFIG['shapefile'] and os.path.exists(CONFIG['shapefile']):
        clip_raster_by_shapefile(temp_output, CONFIG['shapefile'], CONFIG['output_file_raw'])
        if os.path.exists(temp_output):
            os.remove(temp_output)
    else:
        if os.path.exists(temp_output):
            os.rename(temp_output, CONFIG['output_file_raw'])

    # BƯỚC 8: HẬU XỬ LÝ
    print("\n6️⃣  HẬU XỬ LÝ THỦY LỰC (CAP-ONLY)...")
    intermediate_file = CONFIG['output_file'].replace('.tif', '_AFTER_CAP.tif')
    hydraulic_post_correction(
        depth_path=CONFIG['output_file_raw'],
        hand_path=TIFF_PATHS.get('hand30_100'),
        gsw_path=TIFF_PATHS.get('gsw_occure'),
        distance_r_path=TIFF_PATHS.get('distance_r'),
        flow_accum_path=TIFF_PATHS.get('flow_accum'),
        output_path=intermediate_file,
        river_max_wse=4.0, lagoon_max_wse=2.5,
        coastal_max_wse=2.5, inland_max_wse=4.0,
        gsw_threshold=50, river_dist_threshold=50,
        river_hand_max=5.0, lagoon_dist_threshold=500,
        coastal_hand_max=10.0, coastal_dist_min=1000.0,
        flow_accum_percentile=95,
        apply_smoothing=True, smoothing_sigma=0.7,
    )

    # BƯỚC 9: BATHTUB
    print("\n7️⃣  BATHTUB SPREADING...")
    floodplain_bathtub_spreading(
        depth_path=intermediate_file,
        dem_path=TIFF_PATHS.get('dsm_bathy'),
        hand_path=TIFF_PATHS.get('hand30_100'),
        gsw_path=TIFF_PATHS.get('gsw_occure'),
        output_path=CONFIG['output_file'],
        fountain_min_depth=0.8, fountain_max_hand=2.0,
        wse_attenuation_per_km=1.0,
        max_spread_distance=10000,
        floodplain_max_hand=8.0,
        pixel_size_m=30,
        final_clip_depth=3.5,  # ⭐ v6: hạ 5.0 → 3.5
    )

    if os.path.exists(intermediate_file):
        os.remove(intermediate_file)

    # BƯỚC 10: PHÂN LOẠI & CSV
    print("\n8️⃣  TẠO BẢN ĐỒ PHÂN LOẠI...")
    create_flood_classification_map(CONFIG['output_file'], CONFIG['output_classified'], CONFIG['output_legend'])

    print("\n9️⃣  XUẤT CSV...")
    try:
        with rasterio.open(CONFIG['output_file']) as src:
            depth_data = src.read(1)
            transform = src.transform
            height, width = depth_data.shape
            rows_, cols_ = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            valid_mask = depth_data != NODATA_VAL
            if np.any(valid_mask):
                xs, ys = rasterio.transform.xy(transform, rows_[valid_mask], cols_[valid_mask])
                csv_data = pd.DataFrame({
                    'lon': np.round(xs, 6),
                    'lat': np.round(ys, 6),
                    'predicted_depth_m': np.round(depth_data[valid_mask], 4),
                    'flood_class': classify_flood_depth(depth_data[valid_mask])
                })
                if os.path.exists(CONFIG['output_csv']):
                    try:
                        os.remove(CONFIG['output_csv'])
                    except:
                        pass
                csv_data.to_csv(CONFIG['output_csv'], index=False)
                print(f"   ✅ {len(csv_data):,} điểm | Mean: {csv_data['predicted_depth_m'].mean():.4f}m | "
                      f"Max: {csv_data['predicted_depth_m'].max():.4f}m")
    except Exception as e:
        print(f"   ❌ Lỗi CSV: {e}")

    print("\n" + "=" * 78)
    print(f"🎉 HOÀN TẤT - DNN v6 (Sigmoid Output + Final Clip 3.5m)!")
    print("=" * 78)
    print(f"   📁 Manning raster:           {CONFIG['manning_raster']}")
    print(f"   📁 Raster RAW:               {CONFIG['output_file_raw']}")
    print(f"   📁 Raster CHÍNH:             {CONFIG['output_file']} ⭐")
    print(f"   📁 Raster phân loại:         {CONFIG['output_classified']}")
    print(f"   📁 Chú giải:                 {CONFIG['output_legend']}")
    print(f"   📁 CSV:                      {CONFIG['output_csv']}")
    print(f"   📁 Model (Keras):            {CONFIG['model_file']}")
    print(f"   📁 Scaler:                   {CONFIG['scaler_file']}")
    print(f"   📁 Báo cáo:                  {CONFIG['evaluation_report']}")
    print(f"   📁 Training history:         {CONFIG['training_history_plot']}")
    print(f"   📁 Scatter plot:             {CONFIG['scatter_plot']}")
    print(f"   📁 Residuals:                {CONFIG['residuals_plot']}")
    print(f"   📁 Permutation importance:   {CONFIG['permutation_plot']}")


if __name__ == "__main__":
    main()
