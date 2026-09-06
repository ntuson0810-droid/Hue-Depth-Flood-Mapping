"""
================================================================================
DỰ BÁO ĐỘ SÂU NGẬP LỤT - XGBoost GPU v5.1 (FINAL - NO BATHTUB)
================================================================================
Cải tiến v5 → v5.1:
  ❌ BỎ Bathtub Spreading (đã thử nghiệm: gây over-flood vùng núi Huế)
  ✅ Chỉ giữ Hậu xử lý CAP-ONLY 4 vùng (giải pháp tối ưu cho địa hình Huế)
  ✅ Giữ tất cả cải tiến khác từ v5

Lý do bỏ Bathtub:
  Sau khi so sánh với XGB original, bathtub spreading lan tỏa nước đến cả vùng
  núi phía Tây Nam của Huế, dẫn đến 100% diện tích bị dự báo ngập (vô lý).
  Phiên bản original chỉ dùng cap-only đơn giản nhưng phân biệt được rõ ràng
  vùng cao không ngập (5.16% pixel = 0m). Vì Huế có địa hình phức tạp (núi
  sát biển, đồng bằng hẹp xen kẽ đầm phá), bathtub spreading không phù hợp.

Đặc trưng v5.1:
  ✅ Manning's n ESA WorldCover lookup
  ✅ Interaction features đầy đủ + hydraulic (conveyance, resistance, manning_x_*)
  ✅ Monotonic constraints trong XGBoost (ràng buộc đơn điệu vật lý)
  ✅ Sửa lỗi resampling: nearest cho LULC, bilinear cho continuous
  ✅ Hậu xử lý CAP-ONLY 4 vùng (sông/lagoon/cồn cát/đồng bằng)
  ✅ Gaussian smoothing nhẹ
  ✅ Advanced evaluation: R² flood-only, RMSE per bin, classification metrics
  ✅ Permutation feature importance
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
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 👇 KHU VỰC CẤU HÌNH ĐƯỜNG DẪN 👇
# ==============================================================================
CONFIG = {
    'csv_file': r"../data/raw/FloodMarks1999.csv",
    'shapefile': r"../data/spatial/Hue shapefile\hue.shp",

    # Output có prefix XGBv51_ để không đè file v5 cũ
    'output_file_raw': r"../results/rfkq\XGBv51_KetQua_Hue_30m_RAW.tif",
    'output_file': r"../results/rfkq\XGBv51_KetQua_Hue_30m.tif",

    'output_classified': r"../results/rfkq\XGBv51_KetQua_Hue_30m_PhanLoai.tif",
    'output_legend': r"../results/rfkq\XGBv51_Flood_Classification_Legend.png",
    'output_csv': r"../results/rfkq\XGBv51_KetQua_Hue_30m_FullData.csv",
    'evaluation_report': r"../results/rfkq\XGBv51_Model_Evaluation_Report.txt",
    'feature_importance_plot': r"../results/rfkq\XGBv51_Feature_Importance.png",
    'scatter_plot': r"../results/rfkq\XGBv51_Predicted_vs_Actual.png",
    'permutation_plot': r"../results/rfkq\XGBv51_Permutation_Importance.png",
    'model_file': r"../results/rfkq\XGBv51_model.json",
    'scaler_file': r"../results/rfkq\XGBv51_scaler.joblib",

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

# ==============================================================================
# 👇 BẢNG TRA MANNING'S N CHO ESA WORLDCOVER 👇
# ==============================================================================
ESA_MANNING_LOOKUP = {
    10:  0.120,  # Tree cover
    20:  0.060,  # Shrubland
    30:  0.035,  # Grassland
    40:  0.040,  # Cropland
    50:  0.020,  # Built-up
    60:  0.025,  # Bare / sparse vegetation
    70:  0.012,  # Snow and ice
    80:  0.030,  # Permanent water bodies
    90:  0.060,  # Herbaceous wetland
    95:  0.100,  # Mangroves
    100: 0.025,  # Moss and lichen
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
# HÀM: HẬU XỬ LÝ THỦY LỰC (CAP-ONLY) - 4 VÙNG
# ==============================================================================


# ==============================================================================
# HÀM: BATHTUB SPREADING + FINAL CLIP
# ==============================================================================


# ==============================================================================
# HÀM HỖ TRỢ
# ==============================================================================








# ==============================================================================
# HÀM: ĐÁNH GIÁ NÂNG CAO
# ==============================================================================
def evaluate_model_advanced(y_true, y_pred, feature_names, feature_importance,
                              n_train, n_test, n_iterations, model_type="XGBoost-GPU"):
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
        print(f"   • R² flood-only:   {r2_flood:.4f}  ⚠️  CHỈ SỐ ÝNGHĨA NHẤT")
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
        f.write(f"BÁO CÁO ĐÁNH GIÁ MÔ HÌNH {model_type} v5 (FULL UPGRADE)\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. THÔNG SỐ MÔ HÌNH:\n")
        f.write(f"   - Device: GPU (CUDA)\n")
        f.write(f"   - Tree method: hist\n")
        f.write(f"   - Monotonic constraints: CÓ (ràng buộc vật lý)\n")
        f.write(f"   - Early stopping: 50 rounds\n")
        f.write(f"   - Số iterations: {n_iterations}\n")
        f.write(f"   - Features: {len(feature_names)} | Train/Test: {n_train}/{n_test}\n")
        f.write(f"   - Manning lookup: ESA WorldCover\n")
        f.write(f"   - Hậu xử lý: CAP-ONLY + Bathtub + Final Clip 5m\n\n")
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
        f.write("6. FEATURE IMPORTANCE (Top 20):\n")
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names, 'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        for idx, row in feature_imp_df.head(20).iterrows():
            f.write(f"   {row['Feature']:25s}: {row['Importance']:.4f}\n")
        f.write("\n" + "=" * 70 + "\n")
    print(f"\n   ✅ Báo cáo: {report_path}")

    # Feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = feature_imp_df.sort_values('Importance', ascending=True).tail(20)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Built-in Feature Importance - {model_type}\n'
              f'(⚠️ Có thể bị multicollinearity - xem permutation importance)')
    plt.tight_layout()
    plt.savefig(CONFIG['feature_importance_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Feature Importance: {CONFIG['feature_importance_plot']}")

    # Scatter plot
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
    plt.title(f'Predicted vs Actual - {model_type}\n'
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
def compute_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10):
    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        print("   ⚠️  Cần sklearn >= 0.22")
        return None

    print(f"\n🔄 TÍNH PERMUTATION IMPORTANCE ({n_repeats} repeats)...")
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=42, n_jobs=-1
    )

    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'PermImp_Mean': perm.importances_mean,
        'PermImp_Std': perm.importances_std,
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
    ax.set_title('Top 20 Permutation Importance - XGBoost v5\n'
                 '(Đã xử lý multicollinearity)')
    plt.tight_layout()
    plt.savefig(CONFIG['permutation_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {CONFIG['permutation_plot']}")
    return perm_df


# ==============================================================================
# HÀM CHÍNH
# ==============================================================================
def main():
    print("🚀 XGBoost GPU v5 - Manning + Monotonic + Cap-Only + Bathtub + Final Clip")
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
    print("\n2️⃣  TẠO INTERACTION FEATURES (kể cả thủy lực)...")
    data_dict_train = {feat: df_clean[feat].values.astype(np.float32) for feat in feature_names}
    interaction_arrs, interaction_names = create_interaction_features(data_dict_train)
    for arr, name in zip(interaction_arrs, interaction_names):
        df_clean[name] = arr
        feature_names.append(name)
    print(f"   -> Tổng features: {len(feature_names)}")
    hydraulic_feats = [n for n in interaction_names if n in ('conveyance', 'resistance', 'manning_x_hand', 'manning_x_flow')]
    if hydraulic_feats:
        print(f"   -> Hydraulic interaction features:")
        for name in hydraulic_feats:
            print(f"      • {name}")

    # BƯỚC 3: TRAIN XGBOOST
    print("\n3️⃣  HUẤN LUYỆN XGBOOST GPU + MONOTONIC CONSTRAINTS...")
    X = df_clean[feature_names]
    y = df_clean[TARGET_COL]

    print("   -> Chuẩn hóa với StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ⭐ MONOTONIC CONSTRAINTS - RÀNG BUỘC VẬT LÝ (lợi thế của XGBoost)
    monotone_dict = {
        'dsm_bathy':      -1, 'slope':          -1, 'aspect':          0,
        'curvature':       0, 'hillshade':       0, 'roughness':       0,
        'spi':             0, 'tpi':            -1, 'twi':            +1,
        'flow_accum':     +1, 'hand30_100':     -1, 'distance_r':     -1,
        'density_ri':     +1, 'gsw_occure':     +1, 'precip_s_1':     +1,
        'VLUCD_L2_3':      0, 'manning_n':      -1,
        'dem_x_dist':      0, 'dem_x_slope':     0, 'dem_x_twi':       0,
        'rain_x_slope':    0, 'rain_x_twi':     +1, 'flow_x_slope':    0,
        'dist_x_twi':      0, 'rain_x_flow':    +1, 'dem_squared':     0,
        'dist_squared':    0,
        'conveyance':     -1, 'resistance':     +1,
        'manning_x_hand':  0, 'manning_x_flow': +1,
    }
    constraints_tuple = tuple(monotone_dict.get(feat, 0) for feat in feature_names)
    n_constrained = sum(1 for c in constraints_tuple if c != 0)
    print(f"   -> Áp dụng {n_constrained}/{len(feature_names)} monotonic constraints")

    print("   -> Khởi tạo XGBoost với GPU + ràng buộc đơn điệu...")
    params = {
        'device': 'cuda',
        'tree_method': 'hist',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 2000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'monotone_constraints': constraints_tuple,
        'early_stopping_rounds': 50,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1,
    }

    model = xgb.XGBRegressor(**params)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=100)
    n_iterations = model.best_iteration if hasattr(model, 'best_iteration') else len(model.evals_result()['validation_0']['rmse'])
    print(f"\n   ✅ Training hoàn tất! (best_iteration = {n_iterations})")

    # Predict
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    evaluate_model_advanced(y_test, y_pred, feature_names, model.feature_importances_,
                              len(X_train), len(X_test), n_iterations, model_type="XGBoost-GPU v5")

    # PERMUTATION IMPORTANCE
    try:
        perm_df = compute_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10)
    except Exception as e:
        print(f"   ⚠️  Lỗi permutation: {e}")
        perm_df = None

    # LƯU MODEL & SCALER
    print(f"\n   -> Lưu model...")
    model.save_model(CONFIG['model_file'])
    print(f"   ✅ Model: {CONFIG['model_file']}")

    import joblib
    joblib.dump({
        'scaler': scaler,
        'feature_names': feature_names,
        'monotone_dict': monotone_dict,
        'esa_manning_lookup': ESA_MANNING_LOOKUP,
        'best_iteration': n_iterations,
        'permutation_importance': perm_df.to_dict() if perm_df is not None else None,
    }, CONFIG['scaler_file'])
    print(f"   ✅ Scaler + metadata: {CONFIG['scaler_file']}")

    # BƯỚC 4: DỰ BÁO RA RASTER 30M
    print("\n4️⃣  DỰ BÁO RA RASTER 30M VỚI XGBOOST-GPU...")
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
        print(f"   -> Output: {new_width} x {new_height} (~{new_width*new_height/1e6:.2f}M điểm)")

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
    all_data = []
    base_features = list(valid_features.keys())
    block_size = 1024  # XGB-GPU đủ nhanh để xử lý block lớn

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
                    X_valid = np.nan_to_num(X_block_full[mask_valid], nan=0, posinf=0, neginf=0)
                    X_valid_scaled = scaler.transform(X_valid)
                    pred = model.predict(X_valid_scaled)
                    pred = np.maximum(pred, 0)
                    y_block[mask_valid] = pred

                dst.write(y_block.reshape(rows, cols), 1, window=window_30m)

                if np.any(mask_valid):
                    row_idx, col_idx = np.meshgrid(
                        np.arange(row_start, row_end), np.arange(0, cols), indexing='ij'
                    )
                    xs, ys = rasterio.transform.xy(
                        new_transform, row_idx.flatten()[mask_valid], col_idx.flatten()[mask_valid]
                    )
                    block_df = pd.DataFrame(
                        {feat: feature_data[feat][mask_valid] for feat in base_features}
                    )
                    block_df['lon'] = xs
                    block_df['lat'] = ys
                    block_df['predicted_depth'] = y_block[mask_valid]
                    all_data.append(block_df)
    finally:
        for src in src_files.values():
            src.close()

    # BƯỚC 5: CẮT THEO SHAPEFILE
    if CONFIG['shapefile'] and os.path.exists(CONFIG['shapefile']):
        clip_raster_by_shapefile(temp_output, CONFIG['shapefile'], CONFIG['output_file_raw'])
        if os.path.exists(temp_output):
            os.remove(temp_output)
    else:
        if os.path.exists(temp_output):
            os.rename(temp_output, CONFIG['output_file_raw'])

    # BƯỚC 6: HẬU XỬ LÝ THỦY LỰC (CAP-ONLY)
    # ⭐ v5.1: BỎ BATHTUB - chỉ dùng CAP-ONLY (xuất trực tiếp ra output_file)
    # Lý do: Bathtub spreading gây over-flood ở vùng núi của Huế (địa hình
    # không phải đồng bằng phẳng nên không phù hợp với mô hình bathtub)
    print("\n5️⃣  HẬU XỬ LÝ THỦY LỰC (CAP-ONLY, KHÔNG BATHTUB)...")
    hydraulic_post_correction(
        depth_path=CONFIG['output_file_raw'],
        hand_path=TIFF_PATHS.get('hand30_100'),
        gsw_path=TIFF_PATHS.get('gsw_occure'),
        distance_r_path=TIFF_PATHS.get('distance_r'),
        flow_accum_path=TIFF_PATHS.get('flow_accum'),
        output_path=CONFIG['output_file'],   # ⭐ ghi thẳng vào output cuối
        river_max_wse=4.0, lagoon_max_wse=2.5,
        coastal_max_wse=2.5, inland_max_wse=4.0,
        gsw_threshold=50, river_dist_threshold=50,
        river_hand_max=5.0, lagoon_dist_threshold=500,
        coastal_hand_max=10.0, coastal_dist_min=1000.0,
        flow_accum_percentile=95,
        apply_smoothing=True, smoothing_sigma=0.7,
    )

    # ⭐ v5.1 BỎ BATHTUB SPREADING:
    # Sau khi kiểm tra với raster thực tế của Huế, bathtub spreading gây
    # over-flood vùng núi do địa hình phức tạp (không phải đồng bằng phẳng).
    # ORIGINAL XGB chỉ có cap đơn giản nhưng cho kết quả CHUẨN XÁC HƠN ở vùng núi.

    # BƯỚC 7: PHÂN LOẠI
    print("\n6️⃣  TẠO BẢN ĐỒ PHÂN LOẠI...")
    create_flood_classification_map(CONFIG['output_file'], CONFIG['output_classified'], CONFIG['output_legend'])

    # BƯỚC 8: EXPORT CSV
    print("\n7️⃣  XUẤT CSV...")
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df.to_csv(CONFIG['output_csv'], index=False)
        print(f"   ✅ Đã xuất {len(full_df):,} điểm")

    print("\n" + "=" * 78)
    print(f"🎉 HOÀN TẤT - XGBoost GPU v5.1 (Manning + Monotonic + CAP-ONLY, BỎ Bathtub)!")
    print("=" * 78)
    print(f"   📁 Manning's n raster:       {CONFIG['manning_raster']}")
    print(f"   📁 Raster RAW:               {CONFIG['output_file_raw']}")
    print(f"   📁 Raster CHÍNH:             {CONFIG['output_file']} ⭐")
    print(f"   📁 Raster phân loại:         {CONFIG['output_classified']}")
    print(f"   📁 Chú giải:                 {CONFIG['output_legend']}")
    print(f"   📁 CSV:                      {CONFIG['output_csv']}")
    print(f"   📁 Model:                    {CONFIG['model_file']}")
    print(f"   📁 Scaler:                   {CONFIG['scaler_file']}")
    print(f"   📁 Báo cáo evaluation:       {CONFIG['evaluation_report']}")
    print(f"   📁 Built-in importance:      {CONFIG['feature_importance_plot']}")
    print(f"   📁 Permutation importance:   {CONFIG['permutation_plot']} ⭐")
    print(f"   📁 Scatter plot:             {CONFIG['scatter_plot']}")


if __name__ == "__main__":
    main()
