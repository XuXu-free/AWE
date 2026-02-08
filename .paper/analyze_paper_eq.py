import re

file_path = r'C:\Users\23935\.cache\nanochat\knowledge\2501.14576\AWE-SHAREBOP.tex'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Search for the voltage equation components
keywords = ['U_i^{\\text{cell}}', 'U^{rev}', 'r_1', 't_1', 'log', 'ln']

for kw in keywords:
    print(f"\n--- Context for {kw} ---")
    matches = re.finditer(re.escape(kw), content)
    for i, match in enumerate(matches):
        if i > 5: break
        start = max(0, match.start() - 200)
        end = min(len(content), match.end() + 200)
        print(f"...{content[start:end]}...")
