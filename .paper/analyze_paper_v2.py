import re

file_path = r'C:\Users\23935\.cache\nanochat\knowledge\2501.14576\AWE-SHAREBOP.tex'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the start of the Electrochemical section
start_idx = content.find(r'\subsection{Electrochemical and Production Models}')
if start_idx != -1:
    print("Found Electrochemical section. Reading next 3000 chars...")
    print(content[start_idx:start_idx+3000])
else:
    print("Electrochemical section not found.")

# Find the start of the Thermal section (likely under State-Space Models)
start_idx_thermal = content.find(r'temperature')
if start_idx_thermal != -1:
     # try to find a relevant equation block nearby
     pass

# Look for align blocks
aligns = re.findall(r'\\begin{align}(.*?)\\end{align}', content, re.DOTALL)
print(f"\nFound {len(aligns)} align blocks.")
for i, align in enumerate(aligns):
    print(f"\n--- Align Block {i+1} ---")
    print(align[:500]) # Print first 500 chars of each block

