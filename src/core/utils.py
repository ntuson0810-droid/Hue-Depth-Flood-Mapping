import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import pandas as pd
import geopandas as gpd
import os

ESA_MANNING_LOOKUP = {10:0.120, 20:0.060, 30:0.035, 40:0.040, 50:0.020, 60:0.025, 70:0.012, 80:0.030, 90:0.060, 95:0.100, 100:0.025}
DEFAULT_MANNING = 0.040
NODATA_VAL = -9999



def lulc_to_manning(lulc_array):
    manning = np.full(lulc_array.shape, DEFAULT_MANNING, dtype=np.float32)
    for (cls, n_val) in ESA_MANNING_LOOKUP.items():
        manning[(lulc_array == cls)] = n_val
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
    (unique, counts) = np.unique(lulc, return_counts=True)
    print(f"   📊 Phân bố LULC ESA WorldCover và Manning's n:")
    n_matched = 0
    for (cls, cnt) in zip(unique, counts):
        cls_int = int(cls)
        in_table = (cls_int in ESA_MANNING_LOOKUP)
        n_val = ESA_MANNING_LOOKUP.get(cls_int, DEFAULT_MANNING)
        pct = ((cnt / lulc.size) * 100)
        marker = ('✓' if in_table else '❌')
        if in_table:
            n_matched += cnt
        print(f'      Class {cls_int:3d}: n = {n_val:.3f}  ({pct:5.2f}%) {marker}')
    print(f'   📊 Tỉ lệ pixel match ESA: {((n_matched / lulc.size) * 100):.2f}%')
    meta.update(dtype='float32', nodata=(- 9999), compress='lzw', count=1)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(manning, 1)
    print(f'   ✅ Đã tạo: {output_path}')



def create_interaction_features(data_dict, eps=1e-06):
    feats = []
    names = []

    def add(name, arr):
        feats.append(arr.astype(np.float32))
        names.append(name)
    if (('dsm_bathy' in data_dict) and ('distance_r' in data_dict)):
        add('dem_x_dist', (data_dict['dsm_bathy'] * data_dict['distance_r']))
    if (('dsm_bathy' in data_dict) and ('slope' in data_dict)):
        add('dem_x_slope', (data_dict['dsm_bathy'] * data_dict['slope']))
    if (('dsm_bathy' in data_dict) and ('twi' in data_dict)):
        add('dem_x_twi', (data_dict['dsm_bathy'] * data_dict['twi']))
    if (('precip_s_1' in data_dict) and ('slope' in data_dict)):
        add('rain_x_slope', (data_dict['precip_s_1'] * data_dict['slope']))
    if (('precip_s_1' in data_dict) and ('twi' in data_dict)):
        add('rain_x_twi', (data_dict['precip_s_1'] * data_dict['twi']))
    if (('flow_accum' in data_dict) and ('slope' in data_dict)):
        add('flow_x_slope', (data_dict['flow_accum'] * data_dict['slope']))
    if (('distance_r' in data_dict) and ('twi' in data_dict)):
        add('dist_x_twi', (data_dict['distance_r'] * data_dict['twi']))
    if (('precip_s_1' in data_dict) and ('flow_accum' in data_dict)):
        add('rain_x_flow', (data_dict['precip_s_1'] * data_dict['flow_accum']))
    if ('dsm_bathy' in data_dict):
        add('dem_squared', (data_dict['dsm_bathy'] ** 2))
    if ('distance_r' in data_dict):
        add('dist_squared', (data_dict['distance_r'] ** 2))
    if (('manning_n' in data_dict) and ('slope' in data_dict)):
        slope_rad = np.radians(np.clip(data_dict['slope'], 0, 89))
        slope_grade = np.tan(slope_rad)
        n_safe = np.maximum(data_dict['manning_n'], 0.005)
        conveyance = ((1.0 / n_safe) * np.sqrt(np.maximum(slope_grade, eps)))
        add('conveyance', conveyance)
        resistance = (n_safe / np.sqrt(np.maximum(slope_grade, eps)))
        resistance = np.clip(resistance, 0, 100)
        add('resistance', resistance)
    if (('manning_n' in data_dict) and ('hand30_100' in data_dict)):
        add('manning_x_hand', (data_dict['manning_n'] * data_dict['hand30_100']))
    if (('manning_n' in data_dict) and ('flow_accum' in data_dict)):
        add('manning_x_flow', (data_dict['manning_n'] * data_dict['flow_accum']))
    return (feats, names)



def hydraulic_post_correction(depth_path, hand_path, gsw_path, distance_r_path, flow_accum_path, output_path, river_max_wse=4.0, lagoon_max_wse=2.5, coastal_max_wse=2.5, inland_max_wse=4.0, gsw_threshold=50, river_dist_threshold=50, river_hand_max=5.0, lagoon_dist_threshold=500, coastal_hand_max=10.0, coastal_dist_min=1000.0, flow_accum_percentile=95, apply_smoothing=True, smoothing_sigma=0.7):
    print('\n💧 ĐANG HẬU XỬ LÝ THỦY LỰC (CAP-ONLY)...')
    with rasterio.open(depth_path) as src_d:
        depth = src_d.read(1).astype(np.float32)
        meta = src_d.meta.copy()
        (d_transform, d_shape, d_crs) = (src_d.transform, depth.shape, src_d.crs)

    def read_aligned(path, default=0.0, resampling=Resampling.bilinear):
        if ((not path) or (not os.path.exists(path))):
            print(f'   ⚠️  Không tìm thấy {path}')
            return np.full(d_shape, default, dtype=np.float32)
        with rasterio.open(path) as src:
            arr = np.full(d_shape, default, dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=arr, src_transform=src.transform, src_crs=src.crs, dst_transform=d_transform, dst_crs=d_crs, resampling=resampling)
            return arr
    hand = read_aligned(hand_path, default=999.0)
    gsw = read_aligned(gsw_path, default=0.0)
    dist_r = read_aligned(distance_r_path, default=99999.0)
    flow_accum = read_aligned(flow_accum_path, default=0.0)
    valid = (depth != NODATA_VAL)
    n_neg = int(((depth < 0) & valid).sum())
    if (n_neg > 0):
        depth[(valid & (depth < 0))] = 0
        print(f'   • B1: cắt {n_neg:,} pixel có depth âm về 0')
    else:
        print(f'   • B1: không có pixel depth âm')
    if (valid.any() and (flow_accum[valid].size > 0)):
        flow_threshold = np.percentile(flow_accum[valid], flow_accum_percentile)
    else:
        flow_threshold = 0
    river_zone = ((((valid & (gsw > gsw_threshold)) & (dist_r < river_dist_threshold)) & (flow_accum > flow_threshold)) & (hand < river_hand_max))
    lagoon_zone = (((valid & (gsw > gsw_threshold)) & (dist_r > lagoon_dist_threshold)) & (~ river_zone))
    coastal_sandbar = (((((valid & (hand < coastal_hand_max)) & (gsw < 30)) & (dist_r > coastal_dist_min)) & (~ river_zone)) & (~ lagoon_zone))
    inland = (((valid & (~ river_zone)) & (~ lagoon_zone)) & (~ coastal_sandbar))
    print(f'''
   📊 PHÂN VÙNG:''')
    print(f'      Sông:    {int(river_zone.sum()):>10,} pixel')
    print(f'      Lagoon:  {int(lagoon_zone.sum()):>10,} pixel')
    print(f'      Cồn cát: {int(coastal_sandbar.sum()):>10,} pixel')
    print(f'      Đồng bằng: {int(inland.sum()):>10,} pixel')
    print(f'''
   ⛔ CAP DEPTH THEO HAND:''')

    def apply_cap(zone_mask, max_wse, zone_name):
        if (not zone_mask.any()):
            return
        before_mean = depth[zone_mask].mean()
        cap = np.maximum(0, (max_wse - hand[zone_mask]))
        n_capped = int((depth[zone_mask] > cap).sum())
        depth[zone_mask] = np.minimum(depth[zone_mask], cap)
        after_mean = depth[zone_mask].mean()
        print(f'   • {zone_name:<10s}: cap={max_wse}m-HAND | capped {n_capped:,} | mean: {before_mean:.2f}→{after_mean:.2f}m')
    apply_cap(river_zone, river_max_wse, 'Sông')
    apply_cap(lagoon_zone, lagoon_max_wse, 'Lagoon')
    apply_cap(coastal_sandbar, coastal_max_wse, 'Cồn cát')
    if inland.any():
        in_low_hand = (inland & (hand < (inland_max_wse + 2)))
        if in_low_hand.any():
            cap = np.maximum(0, (inland_max_wse - hand[in_low_hand]))
            n_capped = int((depth[in_low_hand] > cap).sum())
            depth[in_low_hand] = np.minimum(depth[in_low_hand], cap)
            print(f'   • Đồng bằng: cap={inland_max_wse}m-HAND | capped {n_capped:,}')
    depth[(valid & (depth < 0))] = 0
    if (apply_smoothing and (smoothing_sigma > 0)):
        try:
            from scipy.ndimage import gaussian_filter
            print(f'''
   🌊 SMOOTHING (sigma={smoothing_sigma})''')
            depth_for_smooth = depth.copy()
            depth_for_smooth[(~ valid)] = 0
            smoothed = gaussian_filter(depth_for_smooth, sigma=smoothing_sigma)
            mask_smooth = gaussian_filter(valid.astype(np.float32), sigma=smoothing_sigma)
            with np.errstate(invalid='ignore', divide='ignore'):
                smoothed = np.where((mask_smooth > 0.01), (smoothed / mask_smooth), depth)
            smoothed[(~ valid)] = NODATA_VAL
            smoothed[(valid & (smoothed < 0))] = 0
            depth = smoothed.astype(np.float32)
            print(f'   ✅ Đã làm mượt')
        except ImportError:
            print(f'   ⚠️  Không có scipy')
    valid_vals = depth[valid]
    print(f'''
   📊 SAU CAP-ONLY: Min={valid_vals.min():.3f}m | Max={valid_vals.max():.3f}m | Mean={valid_vals.mean():.3f}m''')
    meta.update(compress='lzw')
    with rasterio.open(output_path, 'w', **meta) as out:
        out.write(depth, 1)
    print(f'   ✅ Đã lưu: {output_path}')



def floodplain_bathtub_spreading(depth_path, dem_path, hand_path, gsw_path, output_path, fountain_min_depth=0.8, fountain_max_hand=2.0, wse_attenuation_per_km=1.0, max_spread_distance=10000, floodplain_max_hand=8.0, pixel_size_m=30, final_clip_depth=5.0):
    '⭐ FIX BUG: thêm final_clip_depth để loại bỏ pixel bất thường sau bathtub.'
    print('\n🛁 ÁP DỤNG BATHTUB SPREADING...')
    with rasterio.open(depth_path) as src:
        depth = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        (d_transform, d_shape, d_crs) = (src.transform, depth.shape, src.crs)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        print('   ❌ THIẾU scipy - FALLBACK copy')
        meta.update(compress='lzw')
        with rasterio.open(output_path, 'w', **meta) as out:
            out.write(depth, 1)
        return

    def read_aligned(path, default=0.0, resampling=Resampling.bilinear):
        if ((not path) or (not os.path.exists(path))):
            return np.full(d_shape, default, dtype=np.float32)
        with rasterio.open(path) as src:
            arr = np.full(d_shape, default, dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=arr, src_transform=src.transform, src_crs=src.crs, dst_transform=d_transform, dst_crs=d_crs, resampling=resampling)
            return arr
    dem = read_aligned(dem_path, default=0.0)
    hand = read_aligned(hand_path, default=999.0)
    gsw = read_aligned(gsw_path, default=0.0)
    valid = ((depth != NODATA_VAL) & (dem > (- 100)))
    fountain_a = ((valid & (depth > fountain_min_depth)) & (hand < fountain_max_hand))
    fountain_b = ((valid & (gsw > 50)) & (depth > 0.5))
    fountain_mask = (fountain_a | fountain_b)
    n_fountains = int(fountain_mask.sum())
    print(f"   • Số 'fountain': {n_fountains:,} (ML+HAND: {int(fountain_a.sum()):,}, GSW: {int(fountain_b.sum()):,})")
    if (n_fountains == 0):
        print('   ⚠️  Không có fountain - bỏ qua')
        meta.update(compress='lzw')
        with rasterio.open(output_path, 'w', **meta) as out:
            out.write(depth, 1)
        return
    wse_fountain = (dem + depth)
    print('   • Distance transform...')
    (distances_px, indices) = distance_transform_edt((~ fountain_mask), return_indices=True)
    distances_m = (distances_px * pixel_size_m)
    nearest_wse = wse_fountain[(indices[0], indices[1])]
    attenuation = (wse_attenuation_per_km * (distances_m / 1000.0))
    propagated_wse = (nearest_wse - attenuation)
    bathtub_depth = np.maximum(0, (propagated_wse - dem)).astype(np.float32)
    bathtub_depth[(distances_m > max_spread_distance)] = 0
    floodplain_zone = ((valid & (hand < floodplain_max_hand)) & (~ fountain_mask))
    update_mask = (floodplain_zone & (bathtub_depth > depth))
    n_updated = int(update_mask.sum())
    if (n_updated > 0):
        increase_mean = (bathtub_depth[update_mask] - depth[update_mask]).mean()
    else:
        increase_mean = 0
    depth_final = depth.copy()
    depth_final[update_mask] = bathtub_depth[update_mask]
    depth_final[(~ valid)] = NODATA_VAL
    depth_final[(valid & (depth_final < 0))] = 0
    n_above_cap = int((depth_final[valid] > final_clip_depth).sum())
    if (n_above_cap > 0):
        max_before = depth_final[valid].max()
        depth_final[(valid & (depth_final > final_clip_depth))] = final_clip_depth
        print(f'   ⛔ FINAL CLIP {final_clip_depth}m: cắt {n_above_cap:,} pixel | Max: {max_before:.2f}m → {final_clip_depth:.2f}m')
    print(f'   • Cập nhật {n_updated:,} pixel | Tăng TB: {increase_mean:.2f}m')
    valid_vals = depth_final[valid]
    print(f'''
   📊 SAU BATHTUB: Min={valid_vals.min():.3f}m | Max={valid_vals.max():.3f}m | Mean={valid_vals.mean():.3f}m''')
    meta.update(compress='lzw')
    with rasterio.open(output_path, 'w', **meta) as out:
        out.write(depth_final, 1)
    print(f'   ✅ Đã lưu: {output_path}')



def extract_values_at_points(tif_path, lons, lats):
    if ((not tif_path) or (not os.path.exists(tif_path))):
        print(f"⚠️  Bỏ qua: {(os.path.basename(tif_path) if tif_path else 'EMPTY')}")
        return np.zeros(len(lons))
    try:
        with rasterio.open(tif_path) as src:
            coords = list(zip(lons, lats))
            vals = [x[0] for x in src.sample(coords)]
            return np.array(vals)
    except Exception as e:
        print(f'❌ Lỗi đọc {tif_path}: {e}')
        return np.zeros(len(lons))



def clip_raster_by_shapefile(input_tif, shapefile_path, output_tif):
    print('\n✂️  Cắt theo shapefile...')
    try:
        gdf = gpd.read_file(shapefile_path)
        with rasterio.open(input_tif) as src:
            if (str(gdf.crs) != str(src.crs)):
                gdf = gdf.to_crs(src.crs)
            shapes = [feature['geometry'] for (_, feature) in gdf.iterrows()]
            (out_image, out_transform) = rasterio.mask.mask(src, shapes, crop=True, nodata=NODATA_VAL)
            out_meta = src.meta.copy()
        out_meta.update({'driver': 'GTiff', 'height': out_image.shape[1], 'width': out_image.shape[2], 'transform': out_transform, 'nodata': NODATA_VAL, 'compress': 'lzw'})
        if os.path.exists(output_tif):
            try:
                os.remove(output_tif)
            except:
                pass
        with rasterio.open(output_tif, 'w', **out_meta) as dest:
            dest.write(out_image)
        print('   ✅ Đã cắt!')
    except Exception as e:
        print(f'   ❌ Lỗi: {e}')



def classify_flood_depth(depth):
    if isinstance(depth, (int, float)):
        if (depth <= 0):
            return 0
        elif (depth <= 0.5):
            return 1
        elif (depth <= 1.0):
            return 2
        elif (depth <= 1.5):
            return 3
        elif (depth <= 2.0):
            return 4
        else:
            return 5
    else:
        classified = np.zeros_like(depth, dtype=np.uint8)
        classified[(depth > 0)] = 1
        classified[(depth > 0.5)] = 2
        classified[(depth > 1.0)] = 3
        classified[(depth > 1.5)] = 4
        classified[(depth > 2.0)] = 5
        return classified



def create_flood_classification_map(depth_raster_path, output_classified_path, output_legend_path):
    print('\n🗺️  Tạo bản đồ phân loại...')
    try:
        with rasterio.open(depth_raster_path) as src:
            depth_data = src.read(1)
            meta = src.meta.copy()
        classified_data = np.zeros_like(depth_data, dtype=np.uint8)
        mask = (depth_data != NODATA_VAL)
        classified_data[mask] = classify_flood_depth(depth_data[mask])
        classified_data[(~ mask)] = 255
        (unique, counts) = np.unique(classified_data[mask], return_counts=True)
        total_pixels = np.sum(counts)
        class_names = ['Cấp 0: Không ngập (0m)', 'Cấp 1: Ngập nhẹ (0-0.5m)', 'Cấp 2: Ngập trung bình (0.5-1.0m)', 'Cấp 3: Ngập nặng (1.0-1.5m)', 'Cấp 4: Ngập rất nặng (1.5-2.0m)', 'Cấp 5: Ngập đặc biệt nghiêm trọng (>2.0m)']
        print('\n   📊 THỐNG KÊ PHÂN LOẠI:')
        for (cls, count) in zip(unique, counts):
            if (cls < len(class_names)):
                pct = ((count / total_pixels) * 100)
                print(f'   {class_names[int(cls)]}: {count:,} ({pct:.2f}%)')
        meta.update({'dtype': 'uint8', 'nodata': 255, 'compress': 'lzw'})
        if os.path.exists(output_classified_path):
            try:
                os.remove(output_classified_path)
            except:
                pass
        with rasterio.open(output_classified_path, 'w', **meta) as dst:
            dst.write(classified_data, 1)
            colormap = {0: (255, 255, 255), 1: (255, 255, 0), 2: (255, 200, 0), 3: (255, 150, 0), 4: (255, 100, 0), 5: (255, 0, 0), 255: (0, 0, 0)}
            dst.write_colormap(1, colormap)
        print(f'   ✅ Đã lưu: {output_classified_path}')
        (fig, ax) = plt.subplots(figsize=(8, 6))
        colors = ['#FFFFFF', '#FFFF00', '#FFC800', '#FF9600', '#FF6400', '#FF0000']
        for (i, (color, name)) in enumerate(zip(colors, class_names)):
            ax.barh(i, 1, color=color, edgecolor='black', linewidth=2)
            ax.text(1.1, i, name, va='center', fontsize=11, weight='bold')
        ax.set_xlim(0, 3)
        ax.set_ylim((- 0.5), 5.5)
        ax.axis('off')
        ax.set_title('PHÂN LOẠI NGẬP LỤT - XGBoost v5\nThừa Thiên Huế - 1999', fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_legend_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'   ✅ Chú giải: {output_legend_path}')
    except Exception as e:
        print(f'   ❌ Lỗi: {e}')

