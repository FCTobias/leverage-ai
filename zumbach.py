import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def stream_words(filepath, limit=2000000):
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.lower()
            text = re.sub(r'[^a-z0-9\s]', '', text)
            for word in text.split():
                if word:
                    yield word
                    count += 1
                    if count >= limit:
                        return

def build_rank_dict(filepath):
    counts = Counter(stream_words(filepath))
    sorted_vocab = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return {word: rank for rank, (word, count) in enumerate(sorted_vocab)}

def get_returns(filepath, rank_dict, window_size=50, limit=2000000):
    window = []
    x_t = []
    for word in stream_words(filepath, limit):
        r = rank_dict.get(word, len(rank_dict))
        log_r = math.log10(r + 1)
        window.append(log_r)
        if len(window) > window_size:
            window.pop(0)
        if len(window) == window_size:
            x_t.append(sum(window) / window_size)
    return np.diff(x_t)

def calculate_zumbach(rt, max_lag=100):
    zumbach_vals = []
    lags = range(1, max_lag + 1)
    
    for tau in lags:
        term1 = np.mean((rt[tau:]**2) * rt[:-tau])
        term2 = np.mean(rt[tau:] * (rt[:-tau]**2))
        zumbach_vals.append(term1 - term2)
        
    return list(lags), zumbach_vals

def plot_zumbach(lags, zumbach_vals, name, color, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(lags, zumbach_vals, marker='.', linestyle='-', color=color, linewidth=1.5, markersize=4)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    
    ax.set_title(f"Zumbach Effect (Time Reversal Asymmetry): {name}", pad=15, fontweight='bold')
    ax.set_xlabel("Lag $\\tau$ (words)")
    ax.set_ylabel("Asymmetry $Z(\\tau)$")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    datasets = [
        {"name": "Human", "file": "wikisent2.txt", "color": "#4B0082", "out": "09_zumbach_human.png"},
        {"name": "Synthetic", "file": "alpaca_gpt4_full.txt", "color": "#FF8C00", "out": "09_zumbach_synth.png"}
    ]
    
    for config in datasets:
        rank_dict = build_rank_dict(config["file"])
        rt = get_returns(config["file"], rank_dict, window_size=50, limit=2000000)
        
        lags, zumbach_vals = calculate_zumbach(rt, max_lag=100)
        plot_zumbach(lags, zumbach_vals, config["name"], config["color"], config["out"])

if __name__ == "__main__":
    main()