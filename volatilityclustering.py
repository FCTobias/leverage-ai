import re
import math
import numpy as np
import pandas as pd
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

def plot_clustering(rt, name, color, filename):
    windows = [500, 2000, 10000]
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    rt_series = pd.Series(rt)
    
    for i, w in enumerate(windows):
        vol = rt_series.rolling(window=w).std().values
        
        axes[i].plot(vol, color=color, linewidth=0.5, alpha=0.9)
        axes[i].set_title(f"{name} Volatility Time Series (Rolling Window W = {w})", pad=10, fontweight='bold')
        axes[i].set_ylabel("$\\sigma$ (Local)")
        axes[i].grid(True, linestyle=':', alpha=0.6)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        
    axes[-1].set_xlabel("Token Position in Text")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    datasets = [
        {"name": "Human", "file": "wikisent2.txt", "color": "#191970", "out": "04b_volatility_clustering_human.png"},
        {"name": "Synthetic", "file": "alpaca_gpt4_full.txt", "color": "#006400", "out": "04b_volatility_clustering_synth.png"}
    ]
    
    for config in datasets:
        rank_dict = build_rank_dict(config["file"])
        rt = get_returns(config["file"], rank_dict, window_size=50, limit=1750000)
        plot_clustering(rt, config["name"], config["color"], config["out"])

if __name__ == "__main__":
    main()