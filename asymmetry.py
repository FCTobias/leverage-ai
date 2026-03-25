import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import skew

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

def plot_asymmetry(rt, name, color_pos, color_neg, filename):
    pos_returns = rt[rt > 0]
    neg_returns = np.abs(rt[rt < 0])
    
    series_skew = skew(rt)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, max(np.max(pos_returns), np.max(neg_returns)), 100)
    
    ax.hist(pos_returns, bins=bins, density=True, log=True, alpha=0.6, color=color_pos, label='Positive Returns ($R_t > 0$)')
    ax.hist(neg_returns, bins=bins, density=True, log=True, alpha=0.6, color=color_neg, label='Negative Returns ($R_t < 0$ magnitudes)')
    
    ax.set_title(f"Gain/Loss Asymmetry: {name}\nSkewness = {series_skew:.4f}", pad=15, fontweight='bold')
    ax.set_xlabel("Return Magnitude")
    ax.set_ylabel("Density (Log Scale)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    datasets = [
        {"name": "Human", "file": "wikisent2.txt", "c_pos": "#1f77b4", "c_neg": "#d62728", "out": "06_gain_loss_asymmetry_human.png"},
        {"name": "Synthetic", "file": "alpaca_gpt4_full.txt", "c_pos": "#2ca02c", "c_neg": "#d62728", "out": "06_gain_loss_asymmetry_synth.png"}
    ]
    
    for config in datasets:
        rank_dict = build_rank_dict(config["file"])
        rt = get_returns(config["file"], rank_dict, window_size=50, limit=1500000)
        plot_asymmetry(rt, config["name"], config["c_pos"], config["c_neg"], config["out"])

if __name__ == "__main__":
    main()