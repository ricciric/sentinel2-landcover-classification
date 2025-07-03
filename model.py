import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50 # https://torchgeo.readthedocs.io/en/stable/api/models.html
from validation import validate
import matplotlib.pyplot as plt


# Create the PatchDataset class
class PatchDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        image = np.load(image_path)
        mask = np.load(mask_path)
        image = torch.from_numpy(image).to(torch.float32)
        mask = torch.from_numpy(mask).to(torch.float32) # Prova a cambiare in long()
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def main():
    # Load the dataset
    dataset = PatchDataset(
        images_dir='dataset/patches/images/',
        masks_dir='dataset/patches/masks/',
        transform=None
    )

    # Validation dataset
    val_dataset = PatchDataset(
        images_dir='dataset/patches/images/',
        masks_dir='dataset/patches/masks/',
        transform=None
    )

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the UNet model with a ResNet34 backbone pretrained
    model = deeplabv3_resnet50(weights='DEFAULT') # To get the latest weights, use 'DEFAULT'
    # We need to change the last layer to match the number of classes in our dataset
    num_classes = 11
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.to(device)

    # Freezing the backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # Train only the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # Metrics for visualization
    train_losses = []
    val_losses = []
    val_accuracy = []
    val_ious = []


    # Training loop
    epochs = 50
    unfreeze_layer4 = True
    for epoch in range(epochs):
        
 
        # Progressive fine-tuning
        if epoch == 20 and unfreeze_layer4:  # Unfreeze backbone of layer 4 after 10 epochs
            print("Unlocking backbone layers for layer4 fine-tuning...")
            for param in model.backbone.layer4.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)  # nuovo LR

        model.train()
        epoch_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device) # shape [B, 3, 512, 512]
            masks = masks.to(device) # shape [B, 512, 512]
            
            optimizer.zero_grad()
            
            outputs = model(images) # shape [B, num_classes, 512, 512]
            
            loss = criterion(outputs['out'], masks.long()) # The output is a dict, we need to access the 'out' key, because there's an aux output too
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation after each epoch
        val_loss, pixel_accuracy, iou = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracy.append(pixel_accuracy)
        val_ious.append(iou)
        
        # Print progresses
        print(f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Pixel Acc: {pixel_accuracy:.4f} | IoU: {iou:.4f}")

    # Save the model
    model_n = 2
    torch.save(model.state_dict(), f"model_state/finetuned_unet_scl_{model_n}.pth")
    print("Model saved to model_state/finetuned_unet_scl.pth")
    # -------------------------
    #     Curves plotting
    # -------------------------
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1,2,2)
    plt.plot(val_accuracy, label='Pixel Accuracy')
    plt.plot(val_ious, label='IoU')
    
    if unfreeze_layer4:
        plt.axvline(x=10, color='green', linestyle='--', label='Unfreeze Layer4')

    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Validation Metrics')

    plt.show()

if __name__ == "__main__":
    main()