import os
import re

files = [
    'src/dnnmanningv5.py',
    'src/dnnnomannings.py',
    'src/rfmanningsv5.py',
    'src/rfnomannings.py',
    'src/xgbmanningsv5.py',
    'src/xgbnomannings.py'
]

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    cleaned_lines = []
    in_docstring = False
    
    for line in lines:
        stripped = line.strip()
        
        # Bo qua docstring (tam thoi cach don gian)
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                continue # docstring 1 dong
            in_docstring = not in_docstring
            continue
            
        if in_docstring:
            continue
            
        # Bo qua block comments # ========
        if stripped.startswith('# =') and len(stripped) > 10:
            continue
            
        # Bo qua comment dong don kieu mo ta
        if stripped.startswith('#') and 'HAM:' in stripped:
            continue
        if stripped.startswith('#') and 'BU?C' in stripped:
            continue

        cleaned_lines.append(line)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

for f in files:
    if os.path.exists(f):
        clean_file(f)
        print(f"Cleaned {f}")
