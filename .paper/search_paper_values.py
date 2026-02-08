
def search_values():
    path = r"d:\Projects\AWE\.paper\AWE-SHAREBOP.tex"
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Search for lambda definitions
    # Look for "lambda" or "parameter" and numbers
    # Or table content
    
    keywords = ["lambda", "weight", "coefficient", "parameter"]
    
    indices = []
    for kw in keywords:
        start = 0
        while True:
            idx = content.find(kw, start)
            if idx == -1:
                break
            # check if it looks like a definition
            snippet = content[idx:min(idx+100, len(content))]
            if "=" in snippet or "set to" in snippet:
                indices.append((kw, idx))
            start = idx + 1
            
    print(f"Found {len(indices)} potential definitions.")
    
    # Print context
    for kw, idx in indices[:10]: # Print first 10 matches
        start = max(0, idx - 100)
        end = min(len(content), idx + 300)
        print(f"\n--- Context for '{kw}' at {idx} ---")
        print(content[start:end])

    # Also look for the specific lambda names found in the equation
    specific_lambdas = ["prod", "track", "temp", "I", "lye", "c"]
    for lam in specific_lambdas:
        full_lam = f"lambda^{{\\text{{{lam}}}}}"
        print(f"\nSearching for {full_lam}...")
        idx = content.find(full_lam)
        if idx != -1:
             # Look forward for value
             start = idx
             end = min(len(content), idx + 2000)
             print(f"--- Context ---")
             print(content[start:end])

if __name__ == "__main__":
    search_tex_values()
