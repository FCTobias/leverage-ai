import re
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

def get_raw_ranks(filepath, rank_dict, limit=2000000):
    ranks = []
    for word in stream_words(filepath, limit):
        ranks.append(rank_dict.get(word, len(rank_dict)))
    return np.array(ranks)

def calculate_scaling(ranks, step=10000):
    n_vals = np.arange(step, len(ranks) + 1, step)
    vols = np.array([np.std(ranks[:n]) for n in n_vals])
    return n_vals, vols

def plot_scaling(n_vals, vols, name, color, filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].plot(n_vals, vols, color=color, linewidth=1.5)
    axes[0].set_title(f"{name} - Volatility ($\\sigma$) vs. Sample Size", pad=15, fontweight='bold')
    axes[0].set_xlabel("Number of Tokens (N)")
    axes[0].set_ylabel("Volatility")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    axes[1].loglog(n_vals, vols, marker='.', linestyle='--', color=color, markersize=4, linewidth=1.5)
    axes[1].set_title(f"{name} - Log-Log Volatility Scaling", pad=15, fontweight='bold')
    axes[1].set_xlabel("Log(N)")
    axes[1].set_ylabel("Log($\\sigma$)")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    datasets = [
        {"name": "Human", "file": "wikisent2.txt", "color": "#4169E1", "out": "04a_volatility_scaling_human.png"},
        {"name": "Synthetic", "file": "alpaca_gpt4_full.txt", "color": "#2E8B57", "out": "04a_volatility_scaling_synth.png"}
    ]
    
    for config in datasets:
        rank_dict = build_rank_dict(config["file"])
        ranks = get_raw_ranks(config["file"], rank_dict, limit=2000000)
        n_vals, vols = calculate_scaling(ranks, step=10000)
        plot_scaling(n_vals, vols, config["name"], config["color"], config["out"])

if __name__ == "__main__":
    main()