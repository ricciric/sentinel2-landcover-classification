import rasterio 
from rasterio.windows import Window
import numpy as np
import os

# Divide the image into smaller patches
PATCH_SIZE = 512 # Size of the patches to extract

image_path = 'dataset/raw/images/T33TTG_20250628T100051_TCI_20m.jp2'
mask_path = 'dataset/raw/mask/T33TTG_20250628T100051_SCL_20m.jp2'

# Output directory
output_dir = 'dataset/patches'
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

# Patching function
def patcher(image_src, mask_src, patch_size, output_dir):
    width, height = image_src.width, image_src.height
    patch_id = 0
    
    # for loop to scroll the image on windows of size patch_size x patch_size
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            window = Window(left, top, patch_size, patch_size)
            
            # True Color patch
            img_patch = image_src.read(window=window)
            # Mask patch
            mask_patch = mask_src.read(1, window=window)
            
            # Partial patch management
            if img_patch.shape[1] < patch_size or img_patch.shape[2] < patch_size:
                continue
            
            # Normalize ????
            #  img_patch = img_patch.astype(np.float32) / 255.0
            
            # Saving the patches as numpy files
            img_file = f'patch_{patch_id:04d}_image.npy'
            mask_file = f'patch_{patch_id:04d}_mask.npy'
            np.save(os.path.join(output_dir, 'images', img_file), img_patch)
            np.save(os.path.join(output_dir, 'masks', mask_file), mask_patch)
            
            patch_id += 1
    print(f"Total patches created: {patch_id}")
    
# Open the image and mask files
with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
    patcher(img_src, mask_src, PATCH_SIZE, output_dir)   