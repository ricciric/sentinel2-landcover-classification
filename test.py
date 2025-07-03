import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
import matplotlib.pyplot as plt

### Test an image with the fine-tuned model

model = deeplabv3_resnet50(weights=None, num_classes=11, aux_loss=True)  # Assuming 11 classes for SCL
model.classifier[4] = torch.nn.Conv2d(256, 11, kernel_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the model state
state_dict = torch.load("model_state/finetuned_unet_scl.pth", weights_only=True)
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux_classifier.4')} # Filter the aux classifier weights with wrong shape
model.load_state_dict(filtered_state_dict, strict=False) 
model.eval()
model.to(device)

patch_n = '0008'  # Change this to the patch you want to test
patch = np.load(f'dataset/patches/images/patch_{patch_n}_image.npy') # image to test
mask = np.load(f'dataset/patches/masks/patch_{patch_n}_mask.npy') # mask to test
# I don't exactly remember the shape of the patch, but it should be [H, W, C] or [C, H, W]
if patch.shape[2] == 3:  # [H, W, C]
    patch = patch.transpose(2,0,1)
elif patch.shape[0] == 3:  # [C, H, W]
    pass
tensor_patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension and move to device, shape [1, C, H, W]

with torch.no_grad():
    output = model(tensor_patch)['out']  # Forward pass through the model
    pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # Get the predicted class for each pixel
    
# Visualize the prediction

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
patch = patch.transpose(1, 2, 0)  # Convert to [H, W, C] for visualization
plt.imshow(patch)
plt.title("Input Patch")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(mask, cmap='jet')
plt.title("Ground Truth Mask")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(pred, cmap='jet')
plt.title("Predicted Mask")
plt.axis('off')

plt.tight_layout()
plt.show()