import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import kurtosis

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

def get_kurtosis_scaling(rt, max_scale=1000, num_scales=40):
    scales = np.unique(np.logspace(0, math.log10(max_scale), num_scales).astype(int))
    kurt_vals = []
    valid_scales = []
    
    for k in scales:
        n = len(rt)
        if n < k * 10:
            continue
        trunc_n = n - (n % k)
        agg_rt = np.sum(rt[:trunc_n].reshape(-1, k), axis=1)
        
        k_val = kurtosis(agg_rt, fisher=True)
        
        if k_val > 0:
            kurt_vals.append(k_val)
            valid_scales.append(k)
            
    return np.array(valid_scales), np.array(kurt_vals)

def plot_intermittency(scales, kurt_vals, name, color, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(scales, kurt_vals, marker='o', linestyle='-', color=color, markersize=5, linewidth=1.5)
    
    ax.set_title(f"Intermittency (Kurtosis Scaling): {name}", pad=15, fontweight='bold')
    ax.set_xlabel("Aggregation Scale ($\\tau$)")
    ax.set_ylabel("Excess Kurtosis $\\kappa(\\tau)$")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    datasets = [
        {"name": "Human", "file": "wikisent2.txt", "color": "#8B008B", "out": "08_intermittency_human.png"},
        {"name": "Synthetic", "file": "alpaca_gpt4_full.txt", "color": "#B8860B", "out": "08_intermittency_synth.png"}
    ]
    
    for config in datasets:
        rank_dict = build_rank_dict(config["file"])
        rt = get_returns(config["file"], rank_dict, window_size=50, limit=2000000)
        
        scales, kurt_vals = get_kurtosis_scaling(rt, max_scale=1000, num_scales=40)
        plot_intermittency(scales, kurt_vals, config["name"], config["color"], config["out"])

if __name__ == "__main__":
    main()