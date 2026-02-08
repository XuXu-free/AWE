import re

file_path = r'C:\Users\23935\.cache\nanochat\knowledge\2501.14576\AWE-SHAREBOP.tex'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

keywords = ['v_c', 'v_{c}', 'v_\text{c}', 'v_{\text{c}}', 'cooling water', 'heat exchanger']

print(f"Searching in {file_path}...\n")

for kw in keywords:
    matches = list(re.finditer(re.escape(kw), content, re.IGNORECASE))
    if matches:
        print(f"\n--- Found {len(matches)} matches for '{kw}' ---")
        # Print first 3 matches with context
        for i, match in enumerate(matches[:3]):
            start = max(0, match.start() - 300)
            end = min(len(content), match.end() + 300)
            text = content[start:end].replace('\n', ' ')
            print(f"Match {i+1}: ...{text}...")
