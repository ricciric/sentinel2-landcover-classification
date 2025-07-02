import torch
import torch.nn.functional as F

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    iou_numerator = 0
    iou_denominator = 0
    
    with torch.no_grad(): # Disable gradient calculation for validation
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs['out'], masks.long())
            val_loss += loss.item()
            
            # Compute predictions
            preds = torch.argmax(outputs['out'], dim=1)  # shape [B, H, W]
            
            # Pixel accuracy
            correct_pixels += (preds == masks).sum().item()
            total_pixels += torch.numel(masks) # torch.numel: Returns the total number of elements in the input tensor.
            
            # IoU calculation
            intersection = ((preds == masks) & (masks > 0)).sum().item()  # Pixels that are correctly predicted as a class
            union = ((preds > 0) | (masks > 0)).sum().item() # Pixels that are either predicted as a class or are ground truth
            iou_numerator += intersection
            iou_denominator += union
            
    avg_loss = val_loss / len(val_loader)
    pixel_accuracy = correct_pixels / total_pixels
    iou = iou_numerator / (iou_denominator + 1e-6)  # Avoid division by zero
    
    return avg_loss, pixel_accuracy, iou    