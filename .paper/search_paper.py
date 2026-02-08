
def search_tex_file():
    path = r"d:\Projects\AWE\.paper\AWE-SHAREBOP.tex"
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Search for "cost function" or "objective"
    indices = []
    keywords = ["objective function", "cost function", "minimize", "optimization problem", "J_"]
    
    for kw in keywords:
        idx = content.find(kw)
        if idx != -1:
            indices.append((kw, idx))
            
    print(f"Found keywords: {indices}")
    
    # Print context around keywords
    for kw, idx in indices:
        start = max(0, idx - 500)
        end = min(len(content), idx + 2000)
        print(f"\n--- Context for '{kw}' ---")
        print(content[start:end])

if __name__ == "__main__":
    search_tex_file()
