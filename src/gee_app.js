/**
 * Hue Depth Flood Mapping - Google Earth Engine App
 * Author: Nguyen Tu Son
 */

// ==============================================================================
// 1. CẤU HÌNH ASSET ID (BẠN CẦN THAY THẾ CÁC LINK DƯỚI ĐÂY SAU KHI UPLOAD XONG)
// ==============================================================================
// Cách lấy: Vào thẻ Assets -> Click vào file ảnh -> Nút copy (Asset ID)
var ASSETS = {
  // RANDOM FOREST (RFv5)
  'RF_LienTuc': 'projects/ee-ntuson2003nts/assets/RFv5_KetQua_Hue_30m', 
  'RF_PhanLoai': 'projects/ee-ntuson2003nts/assets/RFv5_KetQua_Hue_30m_PhanLoai',
  
  // XGBOOST (XGBv51)
  'XGB_LienTuc': 'projects/ee-ntuson2003nts/assets/XGBv51_KetQua_Hue_30m',
  'XGB_PhanLoai': 'projects/ee-ntuson2003nts/assets/XGBv51_KetQua_Hue_30m_PhanLoai',
  
  // DEEP NEURAL NETWORK (DNNv6)
  'DNN_LienTuc': 'projects/ee-ntuson2003nts/assets/DNNv6_KetQua_Hue_30m',
  'DNN_PhanLoai': 'projects/ee-ntuson2003nts/assets/DNNv6_KetQua_Hue_30m_PhanLoai'
};

// ==============================================================================
// 2. CẤU HÌNH HIỂN THỊ (VISUALIZATION)
// ==============================================================================
// Dải màu cho bản đồ độ sâu liên tục (0 -> 5m)
var visLienTuc = {
  min: 0,
  max: 3,
  palette: ['#ffffff', '#00eaff', '#0072ff', '#0026ff', '#8c00ff', '#ff0000', '#800000']
};

// Dải màu cho bản đồ phân vùng rủi ro 6 cấp
var visPhanLoai = {
  min: 0,
  max: 5,
  palette: [
    '#ffffff', // Cấp 0: Không ngập
    '#7ac4e6', // Cấp 1: Ngập rất nhẹ (0-0.5m)
    '#f5e976', // Cấp 2: Ngập nhẹ (0.5-1m)
    '#ffb84d', // Cấp 3: Ngập trung bình (1-1.5m)
    '#ff4d4d', // Cấp 4: Ngập nặng (1.5-2m)
    '#b30000'  // Cấp 5: Thảm họa (>2m)
  ]
};

var classNames = [
  'Cấp 0 (Không ngập)', 
  'Cấp 1 (Ngập rất nhẹ 0-0.5m)', 
  'Cấp 2 (Ngập nhẹ 0.5-1m)', 
  'Cấp 3 (Ngập trung bình 1-1.5m)', 
  'Cấp 4 (Ngập nặng 1.5-2m)', 
  'Cấp 5 (Thảm họa >2m)'
];

// ==============================================================================
// 3. THIẾT KẾ GIAO DIỆN NGƯỜI DÙNG (UI)
// ==============================================================================
// Xóa map mặc định và tạo UI chính
ui.root.clear();
var mapPanel = ui.Map();
var controlPanel = ui.Panel({style: {width: '350px', padding: '15px'}});
ui.root.add(mapPanel);
ui.root.add(controlPanel);

// Focus bản đồ về Huế
mapPanel.setCenter(107.58, 16.46, 11);
mapPanel.setOptions('SATELLITE');

// --- CÁC THÀNH PHẦN CONTROL PANEL ---
var title = ui.Label('Bản Đồ Ngập Lụt Tỉnh Thừa Thiên Huế', {fontSize: '20px', fontWeight: 'bold', color: '#1d3557'});
var subtitle = ui.Label('Dự báo bằng Học máy tích hợp tri thức vật lý (Physics-Informed Machine Learning)', {fontSize: '13px', color: '#457b9d'});
var authorInfo = ui.Label('Dự án: Hue Depth Flood Mapping - Tác giả: Nguyễn Tư Sơn', {fontSize: '11px', color: '#7f8c8d'});

controlPanel.add(title);
controlPanel.add(subtitle);
controlPanel.add(authorInfo);
controlPanel.add(ui.Label('________________________________________________', {margin: '0 0 10px 0', color: '#bdc3c7'}));

// Dropdown chọn Mô hình
var modelLabel = ui.Label('1. Chọn Mô hình học máy:', {fontWeight: 'bold'});
var modelSelect = ui.Select({
  items: ['Random Forest', 'XGBoost (Physics-Informed)', 'Deep Neural Network'],
  value: 'Random Forest',
  style: {width: '100%'}
});
controlPanel.add(modelLabel).add(modelSelect);

// Dropdown chọn Loại bản đồ
var typeLabel = ui.Label('2. Chọn Loại bản đồ:', {fontWeight: 'bold', margin: '15px 8px 8px 8px'});
var typeSelect = ui.Select({
  items: ['Bản đồ Độ sâu liên tục (m)', 'Bản đồ Phân vùng rủi ro (6 cấp)'],
  value: 'Bản đồ Độ sâu liên tục (m)',
  style: {width: '100%'}
});
controlPanel.add(typeLabel).add(typeSelect);

// ==============================================================================
// 4. LEGEND (BẢNG CHÚ GIẢI)
// ==============================================================================
var legendPanel = ui.Panel({
  style: { position: 'bottom-left', padding: '10px', backgroundColor: 'rgba(255, 255, 255, 0.9)' }
});
mapPanel.add(legendPanel);

function updateLegend(isPhanLoai) {
  legendPanel.clear();
  var legendTitle = ui.Label(isPhanLoai ? 'Mức độ Rủi ro Ngập lụt' : 'Độ sâu ngập (m)', {fontWeight: 'bold', fontSize: '14px', margin: '0 0 10px 0'});
  legendPanel.add(legendTitle);
  
  if (isPhanLoai) {
    for (var i = 0; i < 6; i++) {
      var colorBox = ui.Label('', {backgroundColor: visPhanLoai.palette[i], padding: '8px', margin: '0 0 4px 0'});
      var description = ui.Label(classNames[i], {margin: '0 0 4px 6px', fontSize: '12px'});
      var row = ui.Panel([colorBox, description], ui.Panel.Layout.Flow('horizontal'));
      legendPanel.add(row);
    }
  } else {
    // Gradient legend cho liên tục
    var makeRow = function(color, name) {
      var colorBox = ui.Label('', {backgroundColor: color, padding: '8px', margin: '0 0 4px 0'});
      var description = ui.Label(name, {margin: '0 0 4px 6px', fontSize: '12px'});
      return ui.Panel([colorBox, description], ui.Panel.Layout.Flow('horizontal'));
    };
    legendPanel.add(makeRow(visLienTuc.palette[1], '0.1 - 0.5 m'));
    legendPanel.add(makeRow(visLienTuc.palette[2], '0.5 - 1.0 m'));
    legendPanel.add(makeRow(visLienTuc.palette[3], '1.0 - 1.5 m'));
    legendPanel.add(makeRow(visLienTuc.palette[4], '1.5 - 2.0 m'));
    legendPanel.add(makeRow(visLienTuc.palette[5], '2.0 - 3.0 m'));
    legendPanel.add(makeRow(visLienTuc.palette[6], '> 3.0 m'));
  }
}

// ==============================================================================
// 5. LOGIC HIỂN THỊ BẢN ĐỒ
// ==============================================================================
function updateMap() {
  mapPanel.layers().reset();
  
  var model = modelSelect.getValue();
  var type = typeSelect.getValue();
  var isPhanLoai = (type === 'Bản đồ Phân vùng rủi ro (6 cấp)');
  
  var assetKey = '';
  if (model === 'Random Forest') assetKey = 'RF_';
  else if (model === 'XGBoost (Physics-Informed)') assetKey = 'XGB_';
  else if (model === 'Deep Neural Network') assetKey = 'DNN_';
  
  assetKey += isPhanLoai ? 'PhanLoai' : 'LienTuc';
  
  var assetId = ASSETS[assetKey];
  
  try {
    var image = ee.Image(assetId);
    // Mask các giá trị 0 (không ngập) để nhìn xuyên thấu xuống vệ tinh
    var maskedImage = image.updateMask(image.gt(0));
    
    var visParams = isPhanLoai ? visPhanLoai : visLienTuc;
    mapPanel.addLayer(maskedImage, visParams, model + ' - ' + type);
  } catch(e) {
    print('Đang chờ load Asset hoặc Asset ID chưa chính xác...');
  }
  
  updateLegend(isPhanLoai);
}

// Gắn sự kiện khi thay đổi Dropdown
modelSelect.onChange(updateMap);
typeSelect.onChange(updateMap);

// Khởi tạo hiển thị lần đầu
updateMap();
