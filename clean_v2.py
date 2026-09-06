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

for f in files:
    if not os.path.exists(f): continue
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # Remove block headers # =========
    content = re.sub(r'# ={10,}\n#.*?\n# ={10,}\n', '\n', content)
    content = re.sub(r'# ={10,}\n', '', content)
    
    # Remove the large docstring at the top
    content = re.sub(r'\"\"\"[\s\S]*?\"\"\"\n+', '', content, count=1)
    
    # Remove comments that start with # (but not inline comments)
    # Be careful not to remove all comments, just the noisy ones
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') and ('??' in stripped or 'Bước' in stripped or 'BU?C' in stripped or 'HAM:' in stripped or 'B?NG TRA' in stripped):
            continue
        new_lines.append(line)
        
    with open(f, 'w', encoding='utf-8') as file:
        file.write('\n'.join(new_lines))
    print(f"Cleaned {f}")
