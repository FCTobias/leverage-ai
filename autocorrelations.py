import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

def stream_words(filepath, limit=500000):
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

def get_returns(filepath, rank_dict, window_size=50, limit=500000):
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

def calculate_acf(series, max_lag):
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    res = []
    for lag in range(max_lag + 1):
        c = np.sum((series[:n-lag] - mean) * (series[lag:] - mean)) / n
        res.append(c / var)
    return np.array(res)

def damped_cosine(x, A, gamma, omega):
    return A * np.exp(-gamma * np.abs(x)) * np.cos(omega * x)

def zipfian_law(x, A, alpha):
    res = np.zeros_like(x, dtype=float)
    mask = x != 0
    res[mask] = A / np.power(np.abs(x[mask]), alpha)
    res[~mask] = np.nan
    return res

def plot_empirical(lags, acf_vals, title, color, filename, is_short_term=True):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if is_short_term:
        ax.vlines(lags, 0, acf_vals, color=color, linewidth=2, label='Empirical AC')
    else:
        ax.plot(lags, acf_vals, color=color, linewidth=1.5, alpha=0.8, label='Empirical AC')
        
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(title, pad=15, fontweight='bold')
    ax.legend(loc="upper right")
    
    if not is_short_term:
        ax.set_ylim(-0.05, max(acf_vals) * 1.1)
        
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_fitted(lags, acf_vals, popt, r_squared, title, data_color, fit_func, filename, is_short_term=True):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if is_short_term:
        ax.vlines(lags, 0, acf_vals, color=data_color, linewidth=2, label='Empirical AC', zorder=1)
        title_str = f"{title}\n(A={popt[0]:.2f}, $\\gamma$={popt[1]:.2f}, $\\omega$={popt[2]:.2f})"
        fit_label = f'Damped Cosine Fit ($R^2$={r_squared:.3f})'
    else:
        ax.plot(lags, acf_vals, color=data_color, linewidth=1.5, alpha=0.8, label='Empirical AC', zorder=1)
        title_str = f"{title}\n(A={popt[0]:.2f}, $\\alpha$={popt[1]:.2f})"
        fit_label = f'Zipfian Fit ($R^2$={r_squared:.3f})'

    x_fit = np.linspace(min(lags), max(lags), 2000)
    y_fit = fit_func(x_fit, *popt)
    
    ax.plot(x_fit, y_fit, color='red', linewidth=2, label=fit_label, zorder=2)
    ax.axhline(0, color='black', linewidth=0.5)
    
    ax.set_title(title_str, pad=15, fontweight='bold')
    ax.legend(loc="upper right")
    
    if not is_short_term:
        ax.set_ylim(-0.05, max(acf_vals) * 1.1)
        
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    datasets = [
        {"name": "HUMAN", "file": "wikisent2.txt", "color": "#4169E1"},
        {"name": "SYNTH", "file": "alpaca_gpt4_full.txt", "color": "#2E8B57"}
    ]
    
    st_max = 15
    lt_max = 1000
    
    for config in datasets:
        rank_dict = build_rank_dict(config["file"])
        rt = get_returns(config["file"], rank_dict, window_size=50, limit=200000)
        
        st_acf = calculate_acf(rt, max_lag=st_max)
        st_lags = np.arange(-st_max, st_max + 1)
        st_full_acf = np.concatenate((st_acf[::-1][:-1], st_acf))
        
        plot_empirical(st_lags, st_full_acf, f'{config["name"]} Short-Term AC (Returns)', config["color"], f'03_acf_st_empirical_{config["name"].lower()}.png', True)
        
        pos_st_lags = np.arange(st_max + 1)
        popt_st, _ = curve_fit(damped_cosine, pos_st_lags, st_acf, p0=[1.0, 0.5, 2.5])
        res_st = st_acf - damped_cosine(pos_st_lags, *popt_st)
        r2_st = 1 - (np.sum(res_st**2) / np.sum((st_acf - np.mean(st_acf))**2))
        
        plot_fitted(st_lags, st_full_acf, popt_st, r2_st, f'{config["name"]} Short-Term AC Fit', config["color"], damped_cosine, f'03_acf_st_fitted_{config["name"].lower()}.png', True)
        
        abs_rt = np.abs(rt)
        lt_acf = calculate_acf(abs_rt, max_lag=lt_max)
        lt_lags = np.arange(-lt_max, lt_max + 1)
        lt_full_acf = np.concatenate((lt_acf[::-1][:-1], lt_acf))
        
        plot_empirical(lt_lags, lt_full_acf, f'{config["name"]} Long-Term AC (Absolute Returns)', config["color"], f'03_acf_lt_empirical_{config["name"].lower()}.png', False)
        
        pos_lt_lags = np.arange(1, lt_max + 1)
        popt_lt, _ = curve_fit(lambda x, A, alpha: A / np.power(x, alpha), pos_lt_lags, lt_acf[1:], p0=[0.1, 0.3])
        res_lt = lt_acf[1:] - (popt_lt[0] / np.power(pos_lt_lags, popt_lt[1]))
        r2_lt = 1 - (np.sum(res_lt**2) / np.sum((lt_acf[1:] - np.mean(lt_acf[1:]))**2))
        
        plot_fitted(lt_lags, lt_full_acf, popt_lt, r2_lt, f'{config["name"]} Long-Term AC Fit', config["color"], zipfian_law, f'03_acf_lt_fitted_{config["name"].lower()}.png', False)

if __name__ == "__main__":
    main()