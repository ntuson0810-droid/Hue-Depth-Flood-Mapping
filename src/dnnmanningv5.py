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
def lulc_to_manning(lulc_array):
    manning = np.full(lulc_array.shape, DEFAULT_MANNING, dtype=np.float32)
    for cls, n_val in ESA_MANNING_LOOKUP.items():
        manning[lulc_array == cls] = n_val
    return manning


def create_manning_raster_from_lulc(lulc_path, output_path):
    if os.path.exists(output_path):
        print(f"   ✅ Manning's n raster đã tồn tại: {output_path}")
        return
    print(f"   -> Đang tạo Manning's n raster từ LULC ESA WorldCover...")
    with rasterio.open(lulc_path) as src:
        lulc = src.read(1)
        meta = src.meta.copy()
    manning = lulc_to_manning(lulc)

    unique, counts = np.unique(lulc, return_counts=True)
    print(f"   📊 Phân bố LULC ESA WorldCover và Manning's n:")
    n_matched = 0
    for cls, cnt in zip(unique, counts):
        cls_int = int(cls)
        in_table = cls_int in ESA_MANNING_LOOKUP
        n_val = ESA_MANNING_LOOKUP.get(cls_int, DEFAULT_MANNING)
        pct = cnt / lulc.size * 100
        marker = "✓" if in_table else "❌"
        if in_table:
            n_matched += cnt
        print(f"      Class {cls_int:3d}: n = {n_val:.3f}  ({pct:5.2f}%) {marker}")
    print(f"   📊 Tỉ lệ pixel match ESA: {n_matched/lulc.size*100:.2f}%")

    meta.update(dtype='float32', nodata=-9999, compress='lzw', count=1)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(manning, 1)
    print(f"   ✅ Đã tạo: {output_path}")


# ==============================================================================
# HÀM: INTERACTION FEATURES (DÙNG CẢ TRAIN & PREDICT)
# ==============================================================================
def create_interaction_features(data_dict, eps=1e-6):
    feats = []
    names = []

    def add(name, arr):
        feats.append(arr.astype(np.float32))
        names.append(name)

    if 'dsm_bathy' in data_dict and 'distance_r' in data_dict:
        add('dem_x_dist', data_dict['dsm_bathy'] * data_dict['distance_r'])
    if 'dsm_bathy' in data_dict and 'slope' in data_dict:
        add('dem_x_slope', data_dict['dsm_bathy'] * data_dict['slope'])
    if 'dsm_bathy' in data_dict and 'twi' in data_dict:
        add('dem_x_twi', data_dict['dsm_bathy'] * data_dict['twi'])
    if 'precip_s_1' in data_dict and 'slope' in data_dict:
        add('rain_x_slope', data_dict['precip_s_1'] * data_dict['slope'])
    if 'precip_s_1' in data_dict and 'twi' in data_dict:
        add('rain_x_twi', data_dict['precip_s_1'] * data_dict['twi'])
    if 'flow_accum' in data_dict and 'slope' in data_dict:
        add('flow_x_slope', data_dict['flow_accum'] * data_dict['slope'])
    if 'distance_r' in data_dict and 'twi' in data_dict:
        add('dist_x_twi', data_dict['distance_r'] * data_dict['twi'])
    if 'precip_s_1' in data_dict and 'flow_accum' in data_dict:
        add('rain_x_flow', data_dict['precip_s_1'] * data_dict['flow_accum'])
    if 'dsm_bathy' in data_dict:
        add('dem_squared', data_dict['dsm_bathy'] ** 2)
    if 'distance_r' in data_dict:
        add('dist_squared', data_dict['distance_r'] ** 2)

    # HYDRAULIC INTERACTION FEATURES
    if 'manning_n' in data_dict and 'slope' in data_dict:
        slope_rad = np.radians(np.clip(data_dict['slope'], 0, 89))
        slope_grade = np.tan(slope_rad)
        n_safe = np.maximum(data_dict['manning_n'], 0.005)
        conveyance = (1.0 / n_safe) * np.sqrt(np.maximum(slope_grade, eps))
        add('conveyance', conveyance)
        resistance = n_safe / np.sqrt(np.maximum(slope_grade, eps))
        resistance = np.clip(resistance, 0, 100)
        add('resistance', resistance)

    if 'manning_n' in data_dict and 'hand30_100' in data_dict:
        add('manning_x_hand', data_dict['manning_n'] * data_dict['hand30_100'])
    if 'manning_n' in data_dict and 'flow_accum' in data_dict:
        add('manning_x_flow', data_dict['manning_n'] * data_dict['flow_accum'])

    return feats, names


# ==============================================================================
# HÀM: HẬU XỬ LÝ THỦY LỰC (CAP-ONLY)
# ==============================================================================
def hydraulic_post_correction(depth_path, hand_path, gsw_path,
                               distance_r_path, flow_accum_path,
                               output_path,
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
                               smoothing_sigma=0.7):
    print("\n💧 ĐANG HẬU XỬ LÝ THỦY LỰC (CAP-ONLY)...")

    with rasterio.open(depth_path) as src_d:
        depth = src_d.read(1).astype(np.float32)
        meta = src_d.meta.copy()
        d_transform, d_shape, d_crs = src_d.transform, depth.shape, src_d.crs

    def read_aligned(path, default=0.0, resampling=Resampling.bilinear):
        if not path or not os.path.exists(path):
            print(f"   ⚠️  Không tìm thấy {path}")
            return np.full(d_shape, default, dtype=np.float32)
        with rasterio.open(path) as src:
            arr = np.full(d_shape, default, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1), destination=arr,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=d_transform, dst_crs=d_crs,
                resampling=resampling
            )
            return arr

    hand = read_aligned(hand_path, default=999.0)
    gsw = read_aligned(gsw_path, default=0.0)
    dist_r = read_aligned(distance_r_path, default=99999.0)
    flow_accum = read_aligned(flow_accum_path, default=0.0)

    valid = depth != NODATA_VAL

    # B1: Cắt depth âm
    n_neg = int(((depth < 0) & valid).sum())
    if n_neg > 0:
        depth[valid & (depth < 0)] = 0
        print(f"   • B1: cắt {n_neg:,} pixel có depth âm về 0")
    else:
        print(f"   • B1: không có pixel depth âm")

    # B2: Phân vùng
    if valid.any() and flow_accum[valid].size > 0:
        flow_threshold = np.percentile(flow_accum[valid], flow_accum_percentile)
    else:
        flow_threshold = 0

    river_zone = valid & (gsw > gsw_threshold) & (dist_r < river_dist_threshold) & \
                 (flow_accum > flow_threshold) & (hand < river_hand_max)
    lagoon_zone = valid & (gsw > gsw_threshold) & (dist_r > lagoon_dist_threshold) & ~river_zone
    coastal_sandbar = valid & (hand < coastal_hand_max) & (gsw < 30) & \
                      (dist_r > coastal_dist_min) & ~river_zone & ~lagoon_zone
    inland = valid & ~river_zone & ~lagoon_zone & ~coastal_sandbar

    print(f"\n   📊 PHÂN VÙNG:")
    print(f"      Sông:    {int(river_zone.sum()):>10,} pixel")
    print(f"      Lagoon:  {int(lagoon_zone.sum()):>10,} pixel")
    print(f"      Cồn cát: {int(coastal_sandbar.sum()):>10,} pixel")
    print(f"      Đồng bằng: {int(inland.sum()):>10,} pixel")

    # B3: Cap theo HAND
    print(f"\n   ⛔ CAP DEPTH THEO HAND:")

    def apply_cap(zone_mask, max_wse, zone_name):
        if not zone_mask.any():
            return
        before_mean = depth[zone_mask].mean()
        cap = np.maximum(0, max_wse - hand[zone_mask])
        n_capped = int((depth[zone_mask] > cap).sum())
        depth[zone_mask] = np.minimum(depth[zone_mask], cap)
        after_mean = depth[zone_mask].mean()
        print(f"   • {zone_name:<10s}: cap={max_wse}m-HAND | capped {n_capped:,} | "
              f"mean: {before_mean:.2f}→{after_mean:.2f}m")

    apply_cap(river_zone, river_max_wse, "Sông")
    apply_cap(lagoon_zone, lagoon_max_wse, "Lagoon")
    apply_cap(coastal_sandbar, coastal_max_wse, "Cồn cát")

    if inland.any():
        in_low_hand = inland & (hand < inland_max_wse + 2)
        if in_low_hand.any():
            cap = np.maximum(0, inland_max_wse - hand[in_low_hand])
            n_capped = int((depth[in_low_hand] > cap).sum())
            depth[in_low_hand] = np.minimum(depth[in_low_hand], cap)
            print(f"   • Đồng bằng: cap={inland_max_wse}m-HAND | capped {n_capped:,}")

    depth[valid & (depth < 0)] = 0

    # B4: Gaussian smoothing
    if apply_smoothing and smoothing_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            print(f"\n   🌊 SMOOTHING (sigma={smoothing_sigma})")
            depth_for_smooth = depth.copy()
            depth_for_smooth[~valid] = 0
            smoothed = gaussian_filter(depth_for_smooth, sigma=smoothing_sigma)
            mask_smooth = gaussian_filter(valid.astype(np.float32), sigma=smoothing_sigma)
            with np.errstate(invalid='ignore', divide='ignore'):
                smoothed = np.where(mask_smooth > 0.01, smoothed / mask_smooth, depth)
            smoothed[~valid] = NODATA_VAL
            smoothed[valid & (smoothed < 0)] = 0
            depth = smoothed.astype(np.float32)
            print(f"   ✅ Đã làm mượt")
        except ImportError:
            print(f"   ⚠️  Không có scipy")

    valid_vals = depth[valid]
    print(f"\n   📊 SAU CAP-ONLY: Min={valid_vals.min():.3f}m | Max={valid_vals.max():.3f}m | "
          f"Mean={valid_vals.mean():.3f}m")

    meta.update(compress='lzw')
    with rasterio.open(output_path, 'w', **meta) as out:
        out.write(depth, 1)
    print(f"   ✅ Đã lưu: {output_path}")


# ==============================================================================
# HÀM: BATHTUB SPREADING
# ==============================================================================
def floodplain_bathtub_spreading(depth_path, dem_path, hand_path, gsw_path,
                                  output_path,
                                  fountain_min_depth=0.8,
                                  fountain_max_hand=2.0,
                                  wse_attenuation_per_km=1.0,
                                  max_spread_distance=10000,
                                  floodplain_max_hand=8.0,
                                  pixel_size_m=30,
                                  final_clip_depth=3.5):  # ⭐ v6: hạ 5.0 → 3.5
    print("\n🛁 ÁP DỤNG BATHTUB SPREADING...")

    with rasterio.open(depth_path) as src:
        depth = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        d_transform, d_shape, d_crs = src.transform, depth.shape, src.crs

    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        print("   ❌ THIẾU scipy - FALLBACK copy")
        meta.update(compress='lzw')
        with rasterio.open(output_path, 'w', **meta) as out:
            out.write(depth, 1)
        return

    def read_aligned(path, default=0.0, resampling=Resampling.bilinear):
        if not path or not os.path.exists(path):
            return np.full(d_shape, default, dtype=np.float32)
        with rasterio.open(path) as src:
            arr = np.full(d_shape, default, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1), destination=arr,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=d_transform, dst_crs=d_crs,
                resampling=resampling
            )
            return arr

    dem = read_aligned(dem_path, default=0.0)
    hand = read_aligned(hand_path, default=999.0)
    gsw = read_aligned(gsw_path, default=0.0)

    valid = (depth != NODATA_VAL) & (dem > -100)

    fountain_a = valid & (depth > fountain_min_depth) & (hand < fountain_max_hand)
    fountain_b = valid & (gsw > 50) & (depth > 0.5)
    fountain_mask = fountain_a | fountain_b

    n_fountains = int(fountain_mask.sum())
    print(f"   • Số 'fountain': {n_fountains:,} (ML+HAND: {int(fountain_a.sum()):,}, "
          f"GSW: {int(fountain_b.sum()):,})")

    if n_fountains == 0:
        print("   ⚠️  Không có fountain - bỏ qua")
        meta.update(compress='lzw')
        with rasterio.open(output_path, 'w', **meta) as out:
            out.write(depth, 1)
        return

    wse_fountain = dem + depth
    print("   • Distance transform...")
    distances_px, indices = distance_transform_edt(~fountain_mask, return_indices=True)
    distances_m = distances_px * pixel_size_m

    nearest_wse = wse_fountain[indices[0], indices[1]]
    attenuation = wse_attenuation_per_km * (distances_m / 1000.0)
    propagated_wse = nearest_wse - attenuation

    bathtub_depth = np.maximum(0, propagated_wse - dem).astype(np.float32)
    bathtub_depth[distances_m > max_spread_distance] = 0

    floodplain_zone = valid & (hand < floodplain_max_hand) & ~fountain_mask
    update_mask = floodplain_zone & (bathtub_depth > depth)
    n_updated = int(update_mask.sum())

    if n_updated > 0:
        increase_mean = (bathtub_depth[update_mask] - depth[update_mask]).mean()
    else:
        increase_mean = 0

    depth_final = depth.copy()
    depth_final[update_mask] = bathtub_depth[update_mask]
    depth_final[~valid] = NODATA_VAL
    depth_final[valid & (depth_final < 0)] = 0

    # FINAL CLIP
    n_above_cap = int((depth_final[valid] > final_clip_depth).sum())
    if n_above_cap > 0:
        max_before = depth_final[valid].max()
        depth_final[valid & (depth_final > final_clip_depth)] = final_clip_depth
        print(f"   ⛔ FINAL CLIP {final_clip_depth}m: cắt {n_above_cap:,} pixel | "
              f"Max: {max_before:.2f}m → {final_clip_depth:.2f}m")

    print(f"   • Cập nhật {n_updated:,} pixel | Tăng TB: {increase_mean:.2f}m")

    valid_vals = depth_final[valid]
    print(f"\n   📊 SAU BATHTUB: Min={valid_vals.min():.3f}m | Max={valid_vals.max():.3f}m | "
          f"Mean={valid_vals.mean():.3f}m")

    meta.update(compress='lzw')
    with rasterio.open(output_path, 'w', **meta) as out:
        out.write(depth_final, 1)
    print(f"   ✅ Đã lưu: {output_path}")


# ==============================================================================
# HÀM HỖ TRỢ
# ==============================================================================
def extract_values_at_points(tif_path, lons, lats):
    if not tif_path or not os.path.exists(tif_path):
        return np.zeros(len(lons))
    try:
        with rasterio.open(tif_path) as src:
            coords = list(zip(lons, lats))
            vals = [x[0] for x in src.sample(coords)]
            return np.array(vals)
    except Exception as e:
        print(f"❌ Lỗi đọc {tif_path}: {e}")
        return np.zeros(len(lons))


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


def clip_raster_by_shapefile(input_tif, shapefile_path, output_tif):
    print("\n✂️  Cắt theo shapefile...")
    try:
        gdf = gpd.read_file(shapefile_path)
        with rasterio.open(input_tif) as src:
            if str(gdf.crs) != str(src.crs):
                gdf = gdf.to_crs(src.crs)
            shapes = [feature["geometry"] for _, feature in gdf.iterrows()]
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, nodata=NODATA_VAL)
            out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
            "transform": out_transform, "nodata": NODATA_VAL, "compress": "lzw"
        })
        if os.path.exists(output_tif):
            try:
                os.remove(output_tif)
            except:
                pass
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(out_image)
        print("   ✅ Đã cắt!")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")


def classify_flood_depth(depth):
    if isinstance(depth, (int, float)):
        if depth <= 0: return 0
        elif depth <= 0.5: return 1
        elif depth <= 1.0: return 2
        elif depth <= 1.5: return 3
        elif depth <= 2.0: return 4
        else: return 5
    else:
        classified = np.zeros_like(depth, dtype=np.uint8)
        classified[depth > 0] = 1
        classified[depth > 0.5] = 2
        classified[depth > 1.0] = 3
        classified[depth > 1.5] = 4
        classified[depth > 2.0] = 5
        return classified


def create_flood_classification_map(depth_raster_path, output_classified_path, output_legend_path):
    print("\n🗺️  Tạo bản đồ phân loại...")
    try:
        with rasterio.open(depth_raster_path) as src:
            depth_data = src.read(1)
            meta = src.meta.copy()
        classified_data = np.zeros_like(depth_data, dtype=np.uint8)
        mask = depth_data != NODATA_VAL
        classified_data[mask] = classify_flood_depth(depth_data[mask])
        classified_data[~mask] = 255

        unique, counts = np.unique(classified_data[mask], return_counts=True)
        total_pixels = np.sum(counts)
        class_names = [
            "Cấp 0: Không ngập (0m)",
            "Cấp 1: Ngập nhẹ (0-0.5m)",
            "Cấp 2: Ngập trung bình (0.5-1.0m)",
            "Cấp 3: Ngập nặng (1.0-1.5m)",
            "Cấp 4: Ngập rất nặng (1.5-2.0m)",
            "Cấp 5: Ngập đặc biệt nghiêm trọng (>2.0m)"
        ]
        print("\n   📊 THỐNG KÊ PHÂN LOẠI:")
        for cls, count in zip(unique, counts):
            if cls < len(class_names):
                pct = (count / total_pixels) * 100
                print(f"   {class_names[int(cls)]}: {count:,} ({pct:.2f}%)")

        meta.update({'dtype': 'uint8', 'nodata': 255, 'compress': 'lzw'})
        if os.path.exists(output_classified_path):
            try:
                os.remove(output_classified_path)
            except:
                pass
        with rasterio.open(output_classified_path, 'w', **meta) as dst:
            dst.write(classified_data, 1)
            colormap = {
                0: (255, 255, 255), 1: (255, 255, 0), 2: (255, 200, 0),
                3: (255, 150, 0), 4: (255, 100, 0), 5: (255, 0, 0), 255: (0, 0, 0)
            }
            dst.write_colormap(1, colormap)
        print(f"   ✅ Đã lưu: {output_classified_path}")

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#FFFFFF', '#FFFF00', '#FFC800', '#FF9600', '#FF6400', '#FF0000']
        for i, (color, name) in enumerate(zip(colors, class_names)):
            ax.barh(i, 1, color=color, edgecolor='black', linewidth=2)
            ax.text(1.1, i, name, va='center', fontsize=11, weight='bold')
        ax.set_xlim(0, 3)
        ax.set_ylim(-0.5, 5.5)
        ax.axis('off')
        ax.set_title('PHÂN LOẠI NGẬP LỤT - DNN v6\nThừa Thiên Huế - 1999',
                     fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_legend_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ Chú giải: {output_legend_path}")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")


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
