
import numpy as np

def extract_open_loop_section():
    path = r"d:\Projects\AWE\.paper\AWE-SHAREBOP.tex"
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    targets = ["specific heat capacity"]
    for t in targets:
        idx = content.find(t)
        if idx != -1:
            print(f"\n--- Found '{t}' at {idx} ---")
            print(content[max(0, idx-500):idx+500])

if __name__ == "__main__":
    extract_open_loop_section()
