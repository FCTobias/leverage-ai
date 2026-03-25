import re
import math
from collections import Counter
import matplotlib.pyplot as plt

def stream_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.lower()
            text = re.sub(r'[^a-z0-9\s]', '', text)
            for word in text.split():
                if word:
                    yield word

def build_rank_dict(filepath):
    counts = Counter(stream_words(filepath))
    sorted_vocab = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return {word: rank for rank, (word, count) in enumerate(sorted_vocab)}

def calculate_smoothed_series(filepath, rank_dict, window_size, limit):
    window = []
    series = []
    
    for word in stream_words(filepath):
        if len(series) >= limit:
            break
            
        r = rank_dict.get(word, len(rank_dict))
        log_r = math.log10(r + 1)
        window.append(log_r)
        
        if len(window) > window_size:
            window.pop(0)
            
        if len(window) == window_size:
            x_t = sum(window) / window_size
            series.append(x_t)
            
    return series

def save_plot(series, title, line_color, fill_color, filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(series, color=line_color, linewidth=1.0)
    ax.fill_between(range(len(series)), series, color=fill_color, alpha=0.4)
    
    ax.set_title(title, pad=15, fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    ax.set_ylim(0, max(series) * 1.1 if series else 1)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    datasets = [
        {"name": "Human", "file": "wikisent2.txt", "line_color": "#5b7cfa", "fill_color": "#dbe4ff"},
        {"name": "Synthetic", "file": "alpaca_gpt4_full.txt", "line_color": "#2a9d8f", "fill_color": "#d9f0ed"}
    ]
    
    for config in datasets:
        rank_dict = build_rank_dict(config["file"])
        
        series_macro = calculate_smoothed_series(config["file"], rank_dict, window_size=500, limit=500000)
        save_plot(
            series_macro, 
            f'{config["name"]} - Full Macro (w=500)', 
            config["line_color"], 
            config["fill_color"], 
            f'01_timeseries_{config["name"].lower()}_macro.png'
        )
        
        series_mid = calculate_smoothed_series(config["file"], rank_dict, window_size=100, limit=100000)
        save_plot(
            series_mid, 
            f'{config["name"]} - 100k Mid (w=100)', 
            config["line_color"], 
            config["fill_color"], 
            f'01_timeseries_{config["name"].lower()}_mid.png'
        )
        
        series_micro = calculate_smoothed_series(config["file"], rank_dict, window_size=20, limit=2000)
        save_plot(
            series_micro, 
            f'{config["name"]} - 2k Micro (w=20)', 
            config["line_color"], 
            config["fill_color"], 
            f'01_timeseries_{config["name"].lower()}_micro.png'
        )

if __name__ == "__main__":
    main()