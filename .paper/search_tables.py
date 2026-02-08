
def search_tables():
    path = r"d:\Projects\AWE\.paper\AWE-SHAREBOP.tex"
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    keywords = ["Case Study", "Simulation Setup", "Parameters", "Table"]
    
    indices = []
    for kw in keywords:
        idx = content.find(kw)
        if idx != -1:
            indices.append((kw, idx))
            
    print(f"Found sections: {indices}")
    
    # Look specifically for the weights values
    # e.g. "set as" or "=" near lambda
    
    import re
    # Pattern: lambda^{...} ... number
    # This is hard because of tex formatting
    
    # Let's dump the text around "Table" occurrences, hoping for a parameter table
    
    table_indices = [i for i in range(len(content)) if content.startswith(r"\begin{table}", i)]
    print(f"Found {len(table_indices)} tables.")
    
    for i, idx in enumerate(table_indices):
        end_idx = content.find(r"\end{table}", idx)
        print(f"\n--- Table {i+1} ---")
        print(content[idx:end_idx+12])

if __name__ == "__main__":
    search_simulation_setup()
