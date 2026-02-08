
def search_eq23_start():
    path = r"d:\Projects\AWE\.paper\AWE-SHAREBOP.tex"
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find label eq:obj
    idx = content.find(r"\label{eq:obj}")
    if idx != -1:
        # Search backwards for \begin{align}
        start = content.rfind(r"\begin{align}", 0, idx)
        if start != -1:
            print(f"\n--- Equation 23 (Objective) ---")
            print(content[start:idx+50])

if __name__ == "__main__":
    search_eq23_start()
