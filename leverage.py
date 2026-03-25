import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def stream_words(filepath, limit=1000000):
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

def get_returns(filepath, rank_dict, window_size=50, limit=1000000):
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

def calculate_leverage(rt, vol_window=50, max_lag=100):
    vol = pd.Series(rt).rolling(window=vol_window).std().values
    
    valid_idx = ~np.isnan(vol)
    clean_rt = rt[valid_idx]
    clean_vol = vol[valid_idx]
    
    lags = range(1, max_lag + 1)
    correlations = []
    
    for lag in lags:
        past_returns = clean_rt[:-lag]
        future_vol = clean_vol[lag:]
        corr = np.corrcoef(past_returns, future_vol)[0, 1]
        correlations.append(corr)
        
    return list(lags), correlations

def plot_leverage(lags, correlations, name, color, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(lags, correlations, marker='.', linestyle='-', color=color, linewidth=1.0, markersize=4)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    
    ax.set_title(f"Leverage Effect: Past Return vs Future Volatility ({name})", pad=15, fontweight='bold')
    ax.set_xlabel("Lag (words)")
    ax.set_ylabel("Correlation $L(\\tau)$")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    datasets = [
        {"name": "Human", "file": "wikisent2.txt", "color": "#00008B", "out": "05_anti_leverage_human.png"},
        {"name": "Synthetic", "file": "alpaca_gpt4_full.txt", "color": "#8B0000", "out": "05_anti_leverage_synth.png"}
    ]
    
    for config in datasets:
        rank_dict = build_rank_dict(config["file"])
        rt = get_returns(config["file"], rank_dict, window_size=50, limit=1000000)
        
        lags, correlations = calculate_leverage(rt, vol_window=50, max_lag=100)
        plot_leverage(lags, correlations, config["name"], config["color"], config["out"])

if __name__ == "__main__":
    main()