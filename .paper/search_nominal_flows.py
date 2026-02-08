
def search_nominal_flows():
    path = r"d:\Projects\AWE\.paper\AWE-SHAREBOP.tex"
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Search for "v_{i,\text{lye}}^{\text{0}}" definition or values
    # Look for "nominal lye flow", "initial lye flow", "reference lye flow"
    keywords = [
        "nominal lye flow", 
        "nominal cooling water flow",
        "v_{i,\\text{lye}}^{\\text{0}}",
        "v_{\\text{c}}^{0}",
        "set to",
        "equal to"
    ]
    
    for kw in keywords:
        indices = [i for i in range(len(content)) if content.find(kw, i) == i]
        for idx in indices:
            start = max(0, idx - 300)
            end = min(len(content), idx + 300)
            print(f"\n--- Context for '{kw}' at {idx} ---")
            print(content[start:end])

if __name__ == "__main__":
    search_nominal_flows()
