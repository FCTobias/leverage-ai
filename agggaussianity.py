import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm

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

def aggregate_and_standardize(rt, k):
    n = len(rt)
    trunc_n = n - (n % k)
    agg_rt = np.sum(rt[:trunc_n].reshape(-1, k), axis=1)
    return (agg_rt - np.mean(agg_rt)) / np.std(agg_rt)

def plot_aggregational_gaussianity(rt, name, base_color, filename):
    scales = [1, 5, 20, 100]
    alphas = [0.4, 0.6, 0.8, 1.0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for k, alpha in zip(scales, alphas):
        agg_std = aggregate_and_standardize(rt, k)
        counts, bin_edges = np.histogram(agg_std, bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        valid = counts > 0
        ax.plot(bin_centers[valid], counts[valid], color=base_color, alpha=alpha, linewidth=2, label=f'k = {k}')
        
    x_axis = np.linspace(-6, 6, 1000)
    ax.plot(x_axis, norm.pdf(x_axis), 'k--', linewidth=2, label='Standard Normal $N(0,1)$')
    
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(-6, 6)
    
    ax.set_title(f"Aggregational Gaussianity: {name}", pad=15, fontweight='bold')
    ax.set_xlabel("Standardized Aggregated Return")
    ax.set_ylabel("Density (Log Scale)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    datasets = [
        {"name": "Human", "file": "wikisent2.txt", "color": "#1f77b4", "out": "07_aggregational_gaussianity_human.png"},
        {"name": "Synthetic", "file": "alpaca_gpt4_full.txt", "color": "#2ca02c", "out": "07_aggregational_gaussianity_synth.png"}
    ]
    
    for config in datasets:
        rank_dict = build_rank_dict(config["file"])
        rt = get_returns(config["file"], rank_dict, window_size=50, limit=2000000)
        plot_aggregational_gaussianity(rt, config["name"], config["color"], config["out"])

if __name__ == "__main__":
    main()