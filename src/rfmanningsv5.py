"""
================================================================================
DỰ BÁO ĐỘ SÂU NGẬP LỤT - Random Forest + Manning (ESA) + Cap-Only + Bathtub
Phiên bản v5 - Sửa các bug và nâng cấp đánh giá
================================================================================

Khác biệt so với v4:
  ✅ XÁC NHẬN: File lulc_hue.tif của bạn là ESA WorldCover (đã verify từ raster).
     ESA_MANNING_LOOKUP giữ nguyên - ĐÚNG cho dataset.
     (Tên cột VLUCD_L2_3 trong SHP gốc là MISLEADING - bị giữ tên cũ từ tác giả,
      nhưng raster thực tế là ESA, và code re-sample từ raster nên không sai.)
  ❌ FIX BUG: Bathtub spreading tạo pixel depth=21m (bất thường)
     → Thêm final clip ở 5m (match trần của Flood_1999 trong dataset)
  ✅ THÊM: Evaluate sâu hơn (R² flood-only, RMSE theo cấp độ, classification metrics)
  ✅ THÊM: Permutation feature importance (giải quyết multicollinearity)
  ✅ THÊM: Cảnh báo khi tỉ lệ pixel match ESA lookup < 80%

Các phương pháp giữ nguyên từ v4:
  1. Bảng tra Manning's n cho ESA WorldCover (11 lớp)
  2. Tự tạo raster Manning's n từ LULC
  3. Interaction features thủy lực (conveyance, resistance, manning_x_*)
  4. Sửa lỗi resampling (nearest cho categorical, bilinear cho continuous)
  5. Hậu xử lý CAP-ONLY (không ép cứng)
  6. Bathtub Spreading - lan tỏa mặt nước phẳng (Bates & De Roo 2000)
  7. Gaussian smoothing nhẹ
  8. Post-hoc HAND constraint (bù cho việc RF không có monotonic)
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
from sklearn.ensemble import RandomForestRegressor   # ⭐ THAY XGBOOST
import joblib                                          # ⭐ Lưu RF model
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

    # Output có prefix RFv5_ để không đè file v4
    'output_file_raw': r"../results/rfkq\RFv5_KetQua_Hue_30m_RAW.tif",
    'output_file': r"../results/rfkq\RFv5_KetQua_Hue_30m.tif",

    'output_classified': r"../results/rfkq\RFv5_KetQua_Hue_30m_PhanLoai.tif",
    'output_legend': r"../results/rfkq\RFv5_Flood_Classification_Legend.png",
    'output_csv': r"../results/rfkq\RFv5_KetQua_Hue_30m_FullData.csv",
    'evaluation_report': r"../results/rfkq\RFv5_Model_Evaluation_Report.txt",
    'feature_importance_plot': r"../results/rfkq\RFv5_Feature_Importance.png",
    'scatter_plot': r"../results/rfkq\RFv5_Predicted_vs_Actual.png",
    'permutation_plot': r"../results/rfkq\RFv5_Permutation_Importance.png",
    'model_file': r"../results/rfkq\RFv5_model.joblib",

    # Manning raster v5 (dùng VLUCD lookup mới)
    'manning_raster': r"../data/raster\Manning_N_ThuaThienHue_ESA.tif",
}

# 2. ĐƯỜNG DẪN CÁC FILE TIFF (giữ nguyên)
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

# 3. CẤU HÌNH KHÁC
TARGET_COL = 'Flood_1999'
NODATA_VAL = -9999
TARGET_RESOLUTION = 30

# ==============================================================================
# 👇 BẢNG TRA MANNING'S N CHO ESA WORLDCOVER 👇
# ==============================================================================
# ✅ ĐÃ XÁC NHẬN: File raster lulc_hue.tif của bạn là ESA WorldCover (chuẩn quốc tế)
#
# Phân bố thực tế các classes trong khu vực Thừa Thiên Huế:
#   Class 10 (Tree cover):           75.11%  ← MAJORITY (vùng núi phía Tây)
#   Class 80 (Water bodies):          7.29%  (sông Hương, đầm phá)
#   Class 40 (Cropland):              7.11%  (đồng bằng nông nghiệp)
#   Class 30 (Grassland):             4.92%
#   Class 50 (Built-up):              3.04%  (TP Huế, đô thị)
#   Class 60 (Bare/sparse):           1.74%
#   Class 90 (Herbaceous wetland):    0.76%
#   Class 20 (Shrubland):             0.02%
#   Class 95 (Mangroves):             0.00%
#
# Lưu ý về cột VLUCD_L2_3 trong SHP FloodMarks 1999:
#   Tên cột này MISLEADING - tác giả gốc đặt tên theo VLUCD nhưng raster
#   bạn nhận được là ESA WorldCover. Khi code re-sample từ raster, cột này
#   sẽ chứa ESA classes (10, 20, ..., 95).
#
# Nguồn Manning's n: Chow (1959), Kalyanapu et al. (2009), Papaioannou et al. (2018)
# ==============================================================================
ESA_MANNING_LOOKUP = {
    10:  0.120,  # Tree cover (rừng cây gỗ): 0.10-0.16
    20:  0.060,  # Shrubland (cây bụi): 0.04-0.08
    30:  0.035,  # Grassland (đồng cỏ): 0.03-0.04
    40:  0.040,  # Cropland (đất nông nghiệp/lúa): 0.035-0.05
    50:  0.020,  # Built-up (đất xây dựng/đô thị): 0.013-0.025
    60:  0.025,  # Bare / sparse vegetation (đất trống): 0.02-0.03
    70:  0.012,  # Snow and ice (không có ở Huế nhưng có để phòng)
    80:  0.030,  # Permanent water bodies (sông/hồ): 0.025-0.035
    90:  0.060,  # Herbaceous wetland (đầm lầy cỏ): 0.04-0.08
    95:  0.100,  # Mangroves (rừng ngập mặn): 0.08-0.15
    100: 0.025,  # Moss and lichen
}
DEFAULT_MANNING = 0.040
CATEGORICAL_FEATURES = {'VLUCD_L2_3'}  # Tên cột giữ lại theo SHP gốc (dù raster là ESA)

# ==============================================================================
# 🛑 HẾT PHẦN CẤU HÌNH
# ==============================================================================


# ==============================================================================
# HÀM: TẠO RASTER MANNING'S N TỪ LULC VLUCD (Vietnam Land Use/Cover Database)
# ==============================================================================




# ==============================================================================
# HÀM: HẬU XỬ LÝ THỦY LỰC v3 (CAP-ONLY, KHÔNG ÉP)
# ==============================================================================


# ==============================================================================
# HÀM: BATHTUB SPREADING - LAN TỎA MẶT NƯỚC PHẲNG
# ==============================================================================


# ==============================================================================
# ⭐ HÀM MỚI CHO RF: POST-HOC HAND CONSTRAINT
# ==============================================================================
# Lý do: Random Forest không hỗ trợ monotonic constraints như XGBoost.
# Để bù lại, ta áp một bước "lọc nhẹ" sau khi predict:
#   - Nếu pixel có HAND rất cao (>15m) thì depth phải = 0 (không thể ngập)
#   - Nếu pixel xa sông (>5km) và HAND cao thì depth phải nhỏ
# Đây là một dạng "physical sanity check" cho RF.
# ==============================================================================
def apply_posthoc_hand_constraint(pred, hand_values, dist_r_values,
                                   high_hand_threshold=15.0,
                                   far_dist_threshold=5000.0,
                                   far_hand_threshold=10.0):
    """
    Áp dụng ràng buộc HAND post-hoc cho dự báo RF.
    """
    pred = pred.copy()
    # Pixel HAND quá cao → không ngập
    pred[hand_values > high_hand_threshold] = 0
    # Pixel xa sông + HAND cao → giảm depth
    far_high_mask = (dist_r_values > far_dist_threshold) & (hand_values > far_hand_threshold)
    pred[far_high_mask] = 0
    # Đảm bảo non-negative
    pred = np.maximum(pred, 0)
    return pred


# ==============================================================================
# CÁC HÀM HỖ TRỢ CỐT LÕI (giữ nguyên)
# ==============================================================================








def create_flood_legend(output_path, class_names):
    print(f"   -> Đang tạo chú giải bản đồ...")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#FFFFFF', '#FFFF00', '#FFC800', '#FF9600', '#FF6400', '#FF0000']
    for i, (color, name) in enumerate(zip(colors, class_names)):
        ax.barh(i, 1, color=color, edgecolor='black', linewidth=2)
        ax.text(1.1, i, name, va='center', fontsize=11, weight='bold')
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.5, len(class_names) - 0.5)
    ax.axis('off')
    ax.set_title('PHÂN LOẠI CẤP ĐỘ NGẬP LỤT - Random Forest\nTỉnh Thừa Thiên Huế - 1999',
                 fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Đã lưu chú giải: {output_path}")


def evaluate_model(y_true, y_pred, feature_names, feature_importance, model_type="Random Forest"):
    """
    Đánh giá mô hình với metrics SÂU HƠN:
    - Tổng: R², RMSE, MAE, MAPE
    - Vùng ngập thực (Actual > 0): R², RMSE riêng
    - Per-bin: RMSE từng cấp độ ngập
    - Classification metrics nếu phân loại 6 cấp
    """
    print(f"\n📊 ĐÁNH GIÁ MÔ HÌNH {model_type}:")

    # Chuyển về numpy
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    # ===== METRIC TOÀN BỘ =====
    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)
    mask = y_true_np != 0
    mape = np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100 if np.any(mask) else 0

    print(f"\n   ── METRIC TỔNG (n={len(y_true_np)}) ──")
    print(f"   • R²:    {r2:.4f}")
    print(f"   • RMSE:  {rmse:.4f} m")
    print(f"   • MAE:   {mae:.4f} m")
    print(f"   • MAPE:  {mape:.2f}%")

    # ===== METRIC CHỈ TRÊN VÙNG NGẬP THỰC (Actual > 0) =====
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
        print(f"   • R² flood-only:   {r2_flood:.4f}  ⚠️  (chỉ số ÝNGHĨA NHẤT cho dự báo depth)")
        print(f"   • RMSE flood-only: {rmse_flood:.4f} m")
        print(f"   • MAE flood-only:  {mae_flood:.4f} m")
        print(f"   • Bias (Pred-Actual): {bias_flood:+.4f} m")
        if bias_flood < -0.2:
            print(f"      ⚠️  Model UNDERSHOOT ở vùng ngập (regression toward mean của RF)")
    else:
        r2_flood = rmse_flood = mae_flood = bias_flood = float('nan')
        print(f"\n   ⚠️  Quá ít điểm ngập (n={n_flood}) - không tính được flood-only metrics")

    # ===== RMSE THEO TỪNG CẤP ĐỘ =====
    print(f"\n   ── RMSE THEO CẤP ĐỘ NGẬP ──")
    bins = [(0, 0, 'Không ngập'),
            (0.001, 0.5, '0-0.5m'),
            (0.5, 1.0, '0.5-1m'),
            (1.0, 1.5, '1-1.5m'),
            (1.5, 2.0, '1.5-2m'),
            (2.0, 100, '>2m')]
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

    # ===== CLASSIFICATION METRICS (6 cấp) =====
    try:
        from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
        y_true_cls = classify_flood_depth(y_true_np)
        y_pred_cls = classify_flood_depth(np.maximum(y_pred_np, 0))
        acc = accuracy_score(y_true_cls, y_pred_cls)
        # Accuracy "trong 1 cấp" (predict sai ±1 cấp vẫn coi là đúng)
        within_1 = (np.abs(y_true_cls.astype(int) - y_pred_cls.astype(int)) <= 1).mean()
        print(f"\n   ── CLASSIFICATION METRICS (6 cấp 0-5) ──")
        print(f"   • Accuracy (exact):   {acc:.4f}")
        print(f"   • Accuracy (±1 cấp):  {within_1:.4f}")
    except Exception as e:
        acc = within_1 = float('nan')

    # ===== LƯU BÁO CÁO =====
    report_path = CONFIG['evaluation_report']
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"BÁO CÁO ĐÁNH GIÁ MÔ HÌNH {model_type} v5 (VLUCD + Final Clip)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"1. THÔNG SỐ MÔ HÌNH:\n")
        f.write(f"   - Device: CPU đa luồng (n_jobs=-1)\n")
        f.write(f"   - Số lượng features: {len(feature_names)}\n")
        f.write(f"   - Số điểm test: {len(y_true_np)}\n")
        f.write(f"   - Manning lookup: VLUCD (Vietnam Land Use/Cover Database)\n")
        f.write(f"   - Post-hoc HAND constraint: CÓ\n")
        f.write(f"   - Hậu xử lý: CAP-ONLY + Bathtub Spreading + Final Clip 5m\n\n")

        f.write("2. CHỈ SỐ TỔNG (toàn bộ test set):\n")
        f.write(f"   - R²:    {r2:.4f}\n")
        f.write(f"   - RMSE:  {rmse:.4f} m\n")
        f.write(f"   - MAE:   {mae:.4f} m\n")
        f.write(f"   - MAPE:  {mape:.2f}%\n\n")

        f.write(f"3. CHỈ SỐ CHỈ VÙNG NGẬP (Actual > 0, n={n_flood}):\n")
        f.write(f"   - R² flood-only:   {r2_flood:.4f}\n")
        f.write(f"   - RMSE flood-only: {rmse_flood:.4f} m\n")
        f.write(f"   - MAE flood-only:  {mae_flood:.4f} m\n")
        f.write(f"   - Bias:            {bias_flood:+.4f} m\n\n")

        f.write("4. RMSE THEO CẤP ĐỘ NGẬP:\n")
        for label, n, rmse_b, bias_b in bin_results:
            f.write(f"   - {label:<12s} n={n:>4} | RMSE={rmse_b:.3f}m | Bias={bias_b:+.3f}m\n")
        f.write("\n")

        f.write("5. CLASSIFICATION METRICS (phân loại 6 cấp):\n")
        f.write(f"   - Accuracy (exact):   {acc:.4f}\n")
        f.write(f"   - Accuracy (±1 cấp):  {within_1:.4f}\n\n")

        f.write("6. FEATURE IMPORTANCE (Top 20):\n")
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names, 'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        for idx, row in feature_imp_df.head(20).iterrows():
            f.write(f"   {row['Feature']:25s}: {row['Importance']:.4f}\n")
        f.write("\n" + "=" * 70 + "\n")
    print(f"\n   ✅ Đã lưu báo cáo tại: {report_path}")

    # ===== PLOTS =====
    plt.figure(figsize=(10, 8))
    top_features = feature_imp_df.sort_values('Importance', ascending=True).tail(20)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Built-in Feature Importance - {model_type}\n'
              f'(⚠️ Có thể bị multicollinearity - xem permutation importance)')
    plt.tight_layout()
    plt.savefig(CONFIG['feature_importance_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu Feature Importance: {CONFIG['feature_importance_plot']}")

    # Scatter với 2 màu: ngập vs không ngập
    plt.figure(figsize=(9, 8))
    dry_m = y_true_np == 0
    plt.scatter(y_true_np[dry_m], y_pred_np[dry_m], alpha=0.4, s=12,
                color='gray', label=f'Không ngập (n={dry_m.sum()})')
    plt.scatter(y_true_np[~dry_m], y_pred_np[~dry_m], alpha=0.6, s=20,
                color='steelblue', edgecolor='black', linewidth=0.3,
                label=f'Có ngập (n={(~dry_m).sum()})')
    max_val = max(y_true_np.max(), y_pred_np.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='y = x')
    plt.xlabel('Actual Flood Depth (m)')
    plt.ylabel('Predicted Flood Depth (m)')
    plt.title(f'Predicted vs Actual - {model_type}\n'
              f'R² overall = {r2:.4f} | R² flood-only = {r2_flood:.4f}')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['scatter_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu Scatter Plot: {CONFIG['scatter_plot']}")

    return rmse, mae, r2, mape


# ==============================================================================
# HÀM MỚI v5: PERMUTATION FEATURE IMPORTANCE
# ==============================================================================
# Giải quyết multicollinearity của built-in importance:
# - Built-in importance chia đều cho các features tương quan (vd: dsm_bathy, dem_squared,
#   dem_x_twi đều có importance ~0.13)
# - Permutation importance đo "drop in score khi shuffle 1 feature" → reflect đúng
#   đóng góp riêng biệt của feature đó
# ==============================================================================
def compute_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10):
    """Tính permutation importance để cross-check với built-in."""
    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        print("   ⚠️  Cần sklearn >= 0.22 để dùng permutation_importance")
        return None

    print(f"\n🔄 TÍNH PERMUTATION IMPORTANCE ({n_repeats} repeats)...")
    print("   (sẽ mất 1-3 phút)")
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
    print(f"   (Đã giải quyết multicollinearity của built-in importance)")
    for idx, row in perm_df.head(15).iterrows():
        print(f"   • {row['Feature']:<25s} {row['PermImp_Mean']:+.4f} ± {row['PermImp_Std']:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top = perm_df.sort_values('PermImp_Mean', ascending=True).tail(20)
    ax.barh(top['Feature'], top['PermImp_Mean'], xerr=top['PermImp_Std'],
            color='steelblue', edgecolor='black', error_kw={'ecolor': 'red'})
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Permutation Importance (drop in R²)')
    ax.set_title('Top 20 Permutation Importance - Random Forest v5\n'
                 '(Đã xử lý multicollinearity của built-in importance)')
    plt.tight_layout()
    plt.savefig(CONFIG['permutation_plot'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu Permutation Importance: {CONFIG['permutation_plot']}")

    return perm_df




# ==============================================================================
# HÀM CHÍNH (MAIN)
# ==============================================================================
def main():
    print("🌲 CHƯƠNG TRÌNH DỰ BÁO NGẬP LỤT - Random Forest + Manning + Cap-Only + Bathtub")
    print("=" * 78)

    # Tạo thư mục output
    output_dir = os.path.dirname(CONFIG['output_file'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Tạo thư mục: {output_dir}")

    # ============================================
    # BƯỚC 0: TẠO RASTER MANNING'S N
    # ============================================
    print("\n0️⃣  KHỞI TẠO RASTER MANNING'S N...")
    if os.path.exists(TIFF_PATHS['VLUCD_L2_3']):
        manning_dir = os.path.dirname(CONFIG['manning_raster'])
        if manning_dir and not os.path.exists(manning_dir):
            os.makedirs(manning_dir)
        create_manning_raster_from_lulc(TIFF_PATHS['VLUCD_L2_3'], CONFIG['manning_raster'])
        TIFF_PATHS['manning_n'] = CONFIG['manning_raster']
    else:
        print("   ⚠️  Không tìm thấy LULC, bỏ qua Manning's n.")

    # ============================================
    # BƯỚC 1: LOAD DỮ LIỆU
    # ============================================
    if not os.path.exists(CONFIG['csv_file']):
        print("❌ Lỗi: Không tìm thấy file CSV!")
        return

    print("\n1️⃣  ĐANG XỬ LÝ DỮ LIỆU HUẤN LUYỆN...")
    df = pd.read_csv(CONFIG['csv_file'])

    valid_features = {k: v for k, v in TIFF_PATHS.items() if v and os.path.exists(v)}
    feature_names = list(valid_features.keys())
    print(f"   -> Tìm thấy {len(valid_features)} file Tiff hợp lệ.")

    for feat, path in tqdm(valid_features.items(), desc="   Sampling dữ liệu"):
        df[feat] = extract_values_at_points(path, df['lon_new'], df['lat_new'])

    df_clean = df.dropna(subset=[TARGET_COL]).copy()

    # ============================================
    # BƯỚC 2: FEATURE ENGINEERING
    # ============================================
    print("\n2️⃣  TẠO INTERACTION FEATURES (THỦY LỰC)...")
    data_dict_train = {feat: df_clean[feat].values.astype(np.float32) for feat in feature_names}
    interaction_arrs, interaction_names = create_interaction_features(data_dict_train)

    for arr, name in zip(interaction_arrs, interaction_names):
        df_clean[name] = arr
        feature_names.append(name)

    print(f"   -> Tổng số features: {len(feature_names)}")
    hydraulic_feats = [n for n in interaction_names if n in ('conveyance', 'resistance', 'manning_x_hand', 'manning_x_flow')]
    if hydraulic_feats:
        print(f"   -> Hydraulic interaction features:")
        for name in hydraulic_feats:
            print(f"      • {name}")

    # ============================================
    # BƯỚC 3: HUẤN LUYỆN RANDOM FOREST ⭐
    # ============================================
    print("\n3️⃣  HUẤN LUYỆN RANDOM FOREST (CPU đa luồng)...")
    X = df_clean[feature_names]
    y = df_clean[TARGET_COL]

    print("   -> Chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ⭐ RANDOM FOREST PARAMETERS
    print("   -> Khởi tạo Random Forest...")
    rf_params = {
        'n_estimators': 500,        # 500 cây (đủ ổn định, không cần quá nhiều)
        'max_depth': 20,            # giới hạn độ sâu (RF dễ overfit ở depth=None)
        'min_samples_split': 5,     # tối thiểu 5 samples để split
        'min_samples_leaf': 2,      # tối thiểu 2 samples ở leaf node
        'max_features': 'sqrt',     # sqrt(n_features) - chuẩn cho RF regression
        'bootstrap': True,
        'oob_score': True,          # ⭐ Out-of-bag score - đánh giá free!
        'n_jobs': -1,               # ⭐ DÙNG TẤT CẢ CPU CORES
        'random_state': 42,
        'verbose': 1,
    }

    print(f"   -> Tham số RF:")
    for k, v in rf_params.items():
        print(f"      • {k}: {v}")

    model = RandomForestRegressor(**rf_params)
    print(f"\n   -> Đang training (n_estimators={rf_params['n_estimators']} cây)...")
    model.fit(X_train, y_train)
    print(f"   ✅ Training hoàn tất!")
    print(f"   🎯 Out-of-Bag (OOB) Score: {model.oob_score_:.4f}")

    # Predict
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # đảm bảo non-negative

    evaluate_model(y_test, y_pred, feature_names, model.feature_importances_, "Random Forest")

    # ⭐ V5: Permutation importance (giải quyết multicollinearity của built-in)
    perm_df = compute_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10)

    print(f"\n   -> Đang lưu model (joblib)...")
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'rf_params': rf_params,
        'oob_score': model.oob_score_,
        'esa_manning_lookup': ESA_MANNING_LOOKUP,
        'permutation_importance': perm_df.to_dict() if perm_df is not None else None,
    }, CONFIG['model_file'])
    print(f"   ✅ Đã lưu model: {CONFIG['model_file']}")

    # ============================================
    # BƯỚC 4: DỰ BÁO RA RASTER 30M
    # ============================================
    print("\n4️⃣  DỰ BÁO RA RASTER 30M VỚI RANDOM FOREST...")
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
        print(f"   -> Output: {new_width} x {new_height} pixels (~{new_width*new_height/1e6:.2f}M điểm)")

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

    temp_output = CONFIG['output_file_raw'].replace('.tif', '_temp.tif')
    all_data = []
    base_features = list(valid_features.keys())

    # ⭐ Lưu ý: RF predict CHẬM HƠN XGBoost ~5-10x với block lớn
    # Nên dùng block_size nhỏ hơn để tránh OOM trên CPU
    block_size = 512  # giảm từ 1024 → 512 (RF predict tốn RAM)

    try:
        with rasterio.open(temp_output, 'w', **out_meta) as dst:
            for row_start in tqdm(range(0, new_height, block_size), desc="   Đang dự báo"):
                row_end = min(row_start + block_size, new_height)
                rows = row_end - row_start
                cols = new_width

                window_30m = Window(0, row_start, cols, rows)
                window_10m = Window(0, int(row_start * scale), int(cols * scale), int(rows * scale))

                feature_data = {}
                mask_valid = np.ones((rows * cols), dtype=bool)

                for feat in base_features:
                    try:
                        if feat in CATEGORICAL_FEATURES:
                            resamp = Resampling.nearest
                        else:
                            resamp = Resampling.bilinear

                        data = src_files[feat].read(
                            1, window=window_10m, out_shape=(rows, cols),
                            resampling=resamp, boundless=True, fill_value=0
                        ).flatten().astype(np.float32)

                        feature_data[feat] = data

                        if feat == 'dsm_bathy':
                            nodata = src_files[feat].nodata if src_files[feat].nodata is not None else -9999
                            mask_valid = (data > -100) & (data != nodata)

                    except Exception as e:
                        print(f"\n   ⚠️ Lỗi đọc {feat}: {e}")
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

                    # ⭐ POST-HOC HAND CONSTRAINT cho RF (bù cho việc thiếu monotonic)
                    if 'hand30_100' in feature_data and 'distance_r' in feature_data:
                        hand_valid = feature_data['hand30_100'][mask_valid]
                        dist_valid = feature_data['distance_r'][mask_valid]
                        pred = apply_posthoc_hand_constraint(
                            pred, hand_valid, dist_valid,
                            high_hand_threshold=15.0,
                            far_dist_threshold=5000.0,
                            far_hand_threshold=10.0
                        )

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

    # ============================================
    # BƯỚC 5: CẮT THEO SHAPEFILE
    # ============================================
    if CONFIG['shapefile'] and os.path.exists(CONFIG['shapefile']):
        clip_raster_by_shapefile(temp_output, CONFIG['shapefile'], CONFIG['output_file_raw'])
        if os.path.exists(temp_output):
            os.remove(temp_output)
    else:
        if os.path.exists(temp_output):
            os.rename(temp_output, CONFIG['output_file_raw'])

    # ============================================
    # BƯỚC 6: HẬU XỬ LÝ THỦY LỰC (CAP-ONLY)
    # ============================================
    print("\n5️⃣  HẬU XỬ LÝ THỦY LỰC (CAP-ONLY)...")
    intermediate_file = CONFIG['output_file'].replace('.tif', '_AFTER_CAP.tif')
    hydraulic_post_correction(
        depth_path=CONFIG['output_file_raw'],
        hand_path=TIFF_PATHS.get('hand30_100'),
        gsw_path=TIFF_PATHS.get('gsw_occure'),
        distance_r_path=TIFF_PATHS.get('distance_r'),
        flow_accum_path=TIFF_PATHS.get('flow_accum'),
        output_path=intermediate_file,
        river_max_wse=4.0,
        lagoon_max_wse=2.5,
        coastal_max_wse=2.5,
        inland_max_wse=4.0,
        gsw_threshold=50,
        river_dist_threshold=50,
        river_hand_max=5.0,
        lagoon_dist_threshold=500,
        coastal_hand_max=10.0,
        coastal_dist_min=1000.0,
        flow_accum_percentile=95,
        apply_smoothing=True,
        smoothing_sigma=0.7,
    )

    # ============================================
    # BƯỚC 7: BATHTUB SPREADING
    # ============================================
    print("\n6️⃣  BATHTUB SPREADING (LAN TỎA MẶT NƯỚC RA ĐỒNG BẰNG)...")
    floodplain_bathtub_spreading(
        depth_path=intermediate_file,
        dem_path=TIFF_PATHS.get('dsm_bathy'),
        hand_path=TIFF_PATHS.get('hand30_100'),
        gsw_path=TIFF_PATHS.get('gsw_occure'),
        output_path=CONFIG['output_file'],
        fountain_min_depth=0.8,
        fountain_max_hand=2.0,
        wse_attenuation_per_km=1.0,
        max_spread_distance=10000,
        floodplain_max_hand=8.0,
        pixel_size_m=30,
    )

    if os.path.exists(intermediate_file):
        os.remove(intermediate_file)
        print(f"   🗑️  Đã xóa file trung gian")

    # ============================================
    # BƯỚC 8: PHÂN LOẠI
    # ============================================
    print("\n7️⃣  TẠO BẢN ĐỒ PHÂN LOẠI...")
    create_flood_classification_map(CONFIG['output_file'], CONFIG['output_classified'], CONFIG['output_legend'])

    # ============================================
    # BƯỚC 9: EXPORT CSV
    # ============================================
    print("\n8️⃣  XUẤT CSV...")
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df.to_csv(CONFIG['output_csv'], index=False)
        print(f"   ✅ Đã xuất {len(full_df):,} điểm")

    print("\n" + "=" * 78)
    print(f"🎉 HOÀN TẤT - Random Forest v5 (VLUCD Manning + Cap-Only + Bathtub + Final Clip)!")
    print("=" * 78)
    print(f"   📁 Manning's n raster:       {CONFIG['manning_raster']}")
    print(f"   📁 Raster độ sâu RAW:        {CONFIG['output_file_raw']}")
    print(f"   📁 Raster độ sâu CHÍNH:      {CONFIG['output_file']} ⭐")
    print(f"   📁 Raster phân loại:         {CONFIG['output_classified']}")
    print(f"   📁 Chú giải:                 {CONFIG['output_legend']}")
    print(f"   📁 CSV:                      {CONFIG['output_csv']}")
    print(f"   📁 Model (joblib):           {CONFIG['model_file']}")
    print(f"   📁 Báo cáo evaluation:       {CONFIG['evaluation_report']}")
    print(f"   📁 Built-in importance:      {CONFIG['feature_importance_plot']}")
    print(f"   📁 Permutation importance:   {CONFIG['permutation_plot']} ⭐")
    print(f"   📁 Scatter plot:             {CONFIG['scatter_plot']}")
    print(f"\n   ⚠️  LƯU Ý:")
    print(f"   1. Kiểm tra VLUCD lookup table có đúng với phân loại của thầy bạn không")
    print(f"   2. So sánh built-in vs permutation importance để hiểu vai trò feature thực sự")
    print(f"   3. Báo cáo R² flood-only (không phải chỉ R² overall) để khách quan")


if __name__ == "__main__":
    main()
