import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from pathlib import Path

DATASET_FOLDERS = [
    './inputdata/shapes/1shapedata/0',
    './inputdata/shapes/1shapedata/1',
    './inputdata/shapes/1shapedata/2',
    './inputdata/shapes/1shapedata/3',
    './inputdata/shapes/1shapedata/4',
    './inputdata/shapes/1shapedata/5',
    './inputdata/shapes/1shapedata/6',
    './inputdata/shapes/1shapedata/7',
    './inputdata/shapes/1shapedata/8',
    './inputdata/shapes/1shapedata/9',
    './inputdata/shapes/1shapedata/circle',
    './inputdata/shapes/1shapedata/cube',
    './inputdata/shapes/1shapedata/square',
    './inputdata/shapes/1shapedata/torus',
    './inputdata/shapes/1shapedata/triangle',
]

OUTPUT_FOLDER = './outputdata/output'
NUM_COLORS = 2
DENOISE_FILTER_SIZE = 7
MA_WINDOW = 1000
IMAGES_PER_PAGE = 1

def process_and_plot_batch():
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(exist_ok=True)

    all_entries = []
    for folder_path in DATASET_FOLDERS:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Warning: Folder not found: {folder_path}")
            continue

        files = list(folder.glob('*.png')) + list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg'))
        for f in files:
            all_entries.append((f, folder.name, f.name))

    total_images = len(all_entries)
    if total_images == 0:
        print("No images found. Check your DATASET_FOLDERS paths.")
        return

    print(f"Processing {total_images} images into {int(np.ceil(total_images/IMAGES_PER_PAGE))} files...")

    for i in range(0, total_images, IMAGES_PER_PAGE):
        batch = all_entries[i : i + IMAGES_PER_PAGE]
        page_num = (i // IMAGES_PER_PAGE) + 1

        fig, axs = plt.subplots(len(batch), 5, figsize=(25, 4 * len(batch)))
        axs = np.atleast_2d(axs)

        for row, (img_path, folder_name, file_name) in enumerate(batch):
            display_title = f"{folder_name} - {file_name}"
            raw_img = Image.open(img_path).convert('RGB')
            proc_img = raw_img.filter(ImageFilter.MedianFilter(size=DENOISE_FILTER_SIZE)) \
                              .quantize(colors=NUM_COLORS, method=1).convert('RGB')
            axs[row, 0].imshow(raw_img)
            axs[row, 0].set_title(display_title, fontsize=12, fontweight='bold')
            axs[row, 0].axis('off')
            v = (np.array(proc_img, dtype=np.uint32) << [16, 8, 0]).sum(axis=2)
            _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
            sort_idx = np.argsort(counts)[::-1]
            rank_mapping = np.zeros_like(sort_idx)
            rank_mapping[sort_idx] = np.arange(len(sort_idx))
            rank_img = rank_mapping[inv].reshape(v.shape)
            
            corners = [
                rank_img.flatten(),
                np.fliplr(rank_img).flatten(),
                np.flipud(rank_img).flatten(),
                np.flipud(np.fliplr(rank_img)).flatten()
            ]
            for col_idx, series in enumerate(corners):
                ax = axs[row, col_idx + 1]
                window = min(MA_WINDOW, len(series))
                ma = np.convolve(np.log1p(series), np.ones(window)/window, mode='valid')
                
                ax.plot(ma, color=f'C{col_idx}', linewidth=1.2)
                ax.grid(True, alpha=0.2)

        plt.tight_layout()
        save_name = output_dir / f"analysis_page_{page_num:03d}.png"
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Done. Check the '{OUTPUT_FOLDER}' directory.")

if __name__ == "__main__":
    process_and_plot_batch()