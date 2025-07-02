import numpy as np 
import matplotlib.pyplot as plt
import os

images_dir = 'dataset/patches/images/'
masks_dir = 'dataset/patches/masks/'

patch_id = 28 # Choose which patch to visualize
img_file = f'patch_{patch_id:04d}_image.npy'
mask_file = f'patch_{patch_id:04d}_mask.npy'

image = np.load(os.path.join(images_dir, img_file))
image = image.transpose(1, 2, 0)  # Transpose to (height, width, channels)
mask = np.load(os.path.join(masks_dir, mask_file))

# Matplotlib visualization

plt.figure(figsize=(15,5))

# Mostra immagine True Color
plt.subplot(1,3,1)
plt.imshow(image)
plt.title("True Color")
plt.axis('off')

# Mostra maschera (mappa delle classi)
plt.subplot(1,3,2)
plt.imshow(mask, cmap='jet')
plt.title("Maschera SCL")
plt.axis('off')

# Overlay maschera su immagine
plt.subplot(1,3,3)
plt.imshow(image, alpha=0.8)
plt.imshow(mask, cmap='jet', alpha=0.3)
plt.title("Overlay")
plt.axis('off')

plt.show()

# I colori della maschera differiscono poiché normalizzati sui min-max dei dati e non rappresentano le classi originali.
# Per visualizzare le classi originali, è necessario mappare i valori della maschera ai colori corrispondenti, creando una mappa di colori personalizzata.