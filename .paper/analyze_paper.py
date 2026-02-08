import re

file_path = r'C:\Users\23935\.cache\nanochat\knowledge\2501.14576\AWE-SHAREBOP.tex'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Look for equations
equations = re.findall(r'\\begin{equation}(.*?)\\end{equation}', content, re.DOTALL)
print(f"Found {len(equations)} equations.")

# Look for section headers (even if they are on the same line)
sections = re.findall(r'\\section\{(.*?)\}', content)
print(f"Found sections: {sections}")

# Extract relevant snippets around keywords
keywords = ['Electrochemical', 'Thermal', 'Temperature', 'Voltage', 'Ohmic', 'Activation', 'Balance']
for kw in keywords:
    print(f"\n--- Context for {kw} ---")
    matches = re.finditer(kw, content, re.IGNORECASE)
    for i, match in enumerate(matches):
        if i > 2: break # Limit output
        start = max(0, match.start() - 100)
        end = min(len(content), match.end() + 100)
        print(f"...{content[start:end]}...")

