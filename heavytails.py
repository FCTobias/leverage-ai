import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def stream_words(filepath, limit=5000000):
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

def process_and_plot(filepath, name, color, out_file):
    counts = Counter(stream_words(filepath))
    
    ranks = np.arange(1, len(counts) + 1)
    freqs = np.array([c for wd, c in counts.most_common()])
    
    slope, intercept = np.polyfit(np.log10(ranks), np.log10(freqs), 1)
    alpha = abs(slope)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(ranks, freqs, marker='.', linestyle='none', color=color, markersize=2, alpha=0.7)
    ax.loglog(ranks, (10**intercept) * (ranks**slope), 'k--', linewidth=1.5, label=f'Tail Index $\\alpha$: {alpha:.4f}')
    
    ax.set_title(f"{name} - Log-Log Word Frequency", pad=15, fontweight='bold')
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    process_and_plot("wikisent2.txt", "Human", "#5b7cfa", "02_heavy_tails_freq_human.png")
    process_and_plot("alpaca_gpt4_full.txt", "Synthetic", "#2a9d8f", "02_heavy_tails_freq_synthetic.png")

if __name__ == "__main__":
    main()