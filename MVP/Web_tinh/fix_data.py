"""
Fix NaN values in data.js
"""
import re
from pathlib import Path

data_js_path = Path(__file__).parent / "data.js"

print("Reading data.js...")
with open(data_js_path, 'r', encoding='utf-8') as f:
    content = f.read()

print("Fixing NaN values...")
# Replace NaN with null (JavaScript equivalent)
content = re.sub(r'\bNaN\b', 'null', content)

# Fix string "True" and "False" to boolean
content = re.sub(r': "True"', ': true', content)
content = re.sub(r': "False"', ': false', content)

print("Writing fixed data.js...")
with open(data_js_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed all NaN values and boolean strings in data.js")
