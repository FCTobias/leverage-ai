import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import glob

def calculate_tail_index(image_path, k_fraction=0.1):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        
        pixels = list(img.getdata())
        
        freqs = list(Counter(pixels).values())
        freqs.sort(reverse=True)
        
        X = np.array(freqs, dtype=np.float64)
        n = len(X)
        k = max(2, int(n * k_fraction))
        
        if n <= k + 1 or X[k] <= 0:
            return np.nan
            
        xi_hat = np.mean(np.log(X[:k])) - np.log(X[k])
        
        if xi_hat <= 0:
            return np.nan
            
        return 1.0 / xi_hat
    except Exception:
        return np.nan

def process_folder_list(folder_paths):
    alphas = []
    
    for folder_path in folder_paths:
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
            
        for path in image_paths:
            alpha = calculate_tail_index(path)
            if not np.isnan(alpha):
                alphas.append(alpha)
                
    return np.array(alphas)

def main():
    base_dir = os.path.join('inputdata', 'images')
    datasets = ['imgdataset1']
    
    real_folders = [os.path.join(base_dir, ds, 'REAL') for ds in datasets]
    fake_folders = [os.path.join(base_dir, ds, 'FAKE') for ds in datasets]
    
    real_alphas = process_folder_list(real_folders)
    fake_alphas = process_folder_list(fake_folders)
    
    mu_real = np.mean(real_alphas) if len(real_alphas) > 0 else 0
    mu_fake = np.mean(fake_alphas) if len(fake_alphas) > 0 else 0
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(real_alphas, bins=50, alpha=0.5, label=f'REAL (μ={mu_real:.3f})')
    plt.hist(fake_alphas, bins=50, alpha=0.5, label=f'FAKE (μ={mu_fake:.3f})')
    
    plt.xlabel('Tail Index (α)')
    plt.ylabel('Frequency')
    plt.title('Distribution of α across REAL and FAKE images')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('tail_index_distribution.png')
    plt.show()

if __name__ == '__main__':
    main()