import re

file_path = r'C:\Users\23935\.cache\nanochat\knowledge\2501.14576\AWE-SHAREBOP.tex'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

marker = r'C_{\text{c}} \dfrac{{\text{d}}T_{\text{c,out}}}{\text{d}t}'
idx = content.find(marker)

if idx != -1:
    start = max(0, idx - 500)
    end = min(len(content), idx + 500)
    print(content[start:end])
else:
    # Try fuzzy match if exact fails
    print("Exact marker not found. Trying keyword search nearby...")
    kw = 'T_{\text{c,out}}'
    matches = list(re.finditer(re.escape(kw), content))
    for m in matches[:1]:
         start = max(0, m.start() - 500)
         end = min(len(content), m.end() + 500)
         print(content[start:end])
