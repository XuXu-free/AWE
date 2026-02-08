import re

file_path = r'C:\Users\23935\.cache\nanochat\knowledge\2501.14576\AWE-SHAREBOP.tex'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Look for tables
tables = re.findall(r'\\begin{table}(.*?)\\end{table}', content, re.DOTALL)
print(f"Found {len(tables)} tables.")
for i, table in enumerate(tables):
    print(f"\n--- Table {i+1} ---")
    print(table) # Print the whole table content

# Look for the voltage equation again
# It might be inline math or referenced.
# Search for U_i^{cell} definition
u_cell_matches = re.finditer(r'U_i\^\{\\text\{cell\}\}\s*=', content)
for match in u_cell_matches:
    start = max(0, match.start() - 100)
    end = min(len(content), match.end() + 200)
    print(f"\n--- Voltage Equation Context ---")
    print(content[start:end])

