import os
import random
import matplotlib.pyplot as plt
from PIL import Image


def save_random_images_grid(base_path="Dataset", seed=7):
    random.seed(seed)
    all_image_paths = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)
                all_image_paths.append(full_path)

    num_samples = min(len(all_image_paths), 6)
    if num_samples == 0:
        return

    selected_images = random.sample(all_image_paths, num_samples)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < len(selected_images):
            img_path = selected_images[i]
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    plt.tight_layout()

    plt.savefig("Figures/figure_4.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("Figures/figure_4.png", format="png", dpi=600, bbox_inches="tight")


save_random_images_grid("Dataset")
