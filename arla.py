import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import math
import time
import psutil
from tqdm import tqdm

IMAGE_PATH = "/home/tobiasfc/0arla/4 arla (diffusion)/impression.jpg"
NUM_STEPS = 10

def calculate_svi(image_array):
    v = (image_array[:,:,0] << 16) | (image_array[:,:,1] << 8) | image_array[:,:,2]
    
    counts = np.bincount(v.ravel(), minlength=16777216)
    counts = counts[counts > 0]
    ts = np.sort(counts)[::-1]
    
    N_colors = len(ts)
    if N_colors < 10: 
        return 0.0, counts

    log_counts = np.log(ts)
    cum_log = np.cumsum(log_counts[:-1])
    k_vals = np.arange(1, len(ts))
    
    xi = (cum_log / k_vals) - log_counts[1:]
    alpha = 1.0 / np.maximum(xi, 1e-10)
    
    slopes = np.diff(np.log(alpha)) / np.diff(np.log(k_vals))
    tail_start = int(len(slopes) * 0.8)
    extreme_tail = slopes[tail_start:]
    
    svi = np.sqrt(np.mean(np.diff(extreme_tail)**2)) if len(extreme_tail) > 1 else 0.0
    return svi * np.log1p(N_colors), counts

def apply_svi_quantization(image_array, severity=0.1):
    shift_bits = int(np.ceil(severity * 4))
    return (image_array >> shift_bits) << shift_bits

def white_box_svi_diffusion(image_path, steps, learning_rate=0.1, svi_threshold=5.0):
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss
    start_time = time.time()
    
    target = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
    np.random.seed(42)
    current_state = np.random.uniform(0, 255, target.shape).astype(np.float32)
    
    os.makedirs("diffusion_frames", exist_ok=True)
    step_images = []
    
    peak_mem = start_mem
    
    with tqdm(total=steps, desc="Processing Diffusion Steps") as pbar:
        for t in range(steps):
            current_state += learning_rate * (target - current_state)
            noise_level = 255 * (1 - (t / steps)) * 0.2
            current_state = np.clip(current_state + np.random.normal(0, noise_level, target.shape), 0, 255)
            
            img_uint8 = current_state.astype(np.uint8)
            svi_score, counts = calculate_svi(img_uint8)
            
            if svi_score > svi_threshold:
                severity = min(1.0, (svi_score - svi_threshold) / 10.0)
                img_uint8 = apply_svi_quantization(img_uint8, severity)
                current_state = img_uint8.astype(np.float32)
                
            img_pil = Image.fromarray(img_uint8)
            step_images.append(img_pil)
            img_pil.save(f"diffusion_frames/frame_{t:03d}.png")
            
            current_mem = process.memory_info().rss
            if current_mem > peak_mem:
                peak_mem = current_mem
                
            pbar.update(1)

    step_images[-1].save("final_output.png")
    
    cols = int(math.ceil(math.sqrt(steps)))
    rows = int(math.ceil(steps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if steps > 1 else [axes]
    
    for i in range(len(axes)):
        axes[i].axis('off')
        if i < steps:
            axes[i].imshow(step_images[i])
            axes[i].set_title(f"Step {i+1}")
            
    plt.tight_layout()
    plt.savefig("diffusion_steps_plot.png")
    plt.close()

    end_time = time.time()
    sys_mem = psutil.virtual_memory()
    
    print("\n" + "="*40)
    print("HARDWARE & COMPUTE TELEMETRY")
    print("="*40)
    print(f"Total Execution Time : {end_time - start_time:.2f} seconds")
    print(f"Machine Total RAM    : {sys_mem.total / (1024**3):.2f} GB")
    print(f"Machine Available RAM: {sys_mem.available / (1024**3):.2f} GB")
    print(f"Process Peak Memory  : {peak_mem / (1024**2):.2f} MB")
    print(f"Current CPU Usage    : {psutil.cpu_percent(interval=0.1)}%")
    print("="*40)

if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        white_box_svi_diffusion(IMAGE_PATH, steps=NUM_STEPS, learning_rate=0.08, svi_threshold=8.0)
    else:
        print(f"Error: Could not find the image at '{IMAGE_PATH}'.")