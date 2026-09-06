import ast
import os
import astunparse

files = [
    'src/dnnmanningv5.py',
    'src/dnnnomannings.py',
    'src/rfmanningsv5.py',
    'src/rfnomannings.py',
    'src/xgbmanningsv5.py',
    'src/xgbnomannings.py'
]

functions_to_extract = [
    'lulc_to_manning',
    'create_manning_raster_from_lulc',
    'create_interaction_features',
    'hydraulic_post_correction',
    'floodplain_bathtub_spreading',
    'extract_values_at_points',
    'clip_raster_by_shapefile',
    'classify_flood_depth',
    'create_flood_classification_map'
]

# Extract from first file
first_file = 'src/xgbmanningsv5.py'
with open(first_file, 'r', encoding='utf-8-sig') as f:
    source = f.read()
tree = ast.parse(source)

extracted_nodes = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in functions_to_extract:
        extracted_nodes.append(node)

# Write to src/core/utils.py
core_code = "import numpy as np\nimport rasterio\nfrom rasterio.warp import reproject, Resampling\nimport pandas as pd\nimport geopandas as gpd\nimport os\n\n"
core_code += "ESA_MANNING_LOOKUP = {10:0.120, 20:0.060, 30:0.035, 40:0.040, 50:0.020, 60:0.025, 70:0.012, 80:0.030, 90:0.060, 95:0.100, 100:0.025}\n"
core_code += "DEFAULT_MANNING = 0.040\nNODATA_VAL = -9999\n\n"

for node in extracted_nodes:
    core_code += astunparse.unparse(node) + "\n"

os.makedirs('src/core', exist_ok=True)
with open('src/core/__init__.py', 'w') as f: f.write('')
with open('src/core/utils.py', 'w', encoding='utf-8') as f:
    f.write(core_code)

# Remove these functions from all files based on line numbers
for fpath in files:
    if not os.path.exists(fpath): continue
    with open(fpath, 'r', encoding='utf-8-sig') as f:
        src_lines = f.readlines()
    
    t = ast.parse("".join(src_lines))
    
    # Collect line ranges to delete
    ranges_to_delete = []
    for node in t.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions_to_extract:
            ranges_to_delete.append((node.lineno - 1, node.end_lineno - 1))
            
    # Also find where to insert the import
    import_idx = 0
    for i, line in enumerate(src_lines):
        if line.startswith('import ') or line.startswith('from '):
            import_idx = i
            break
            
    new_lines = []
    for i, line in enumerate(src_lines):
        # check if in range
        in_range = False
        for start, end in ranges_to_delete:
            if start <= i <= end:
                in_range = True
                break
        if not in_range:
            new_lines.append(line)
            if i == import_idx:
                new_lines.append("import sys\nimport os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nfrom src.core.utils import *\n")
                
    with open(fpath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Refactored {fpath}")

