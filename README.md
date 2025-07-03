# Land Cover Classification on Sentinel-2 SCL with DeepLabv3

This project performs semantic segmentation on Sentinel-2 satellite imagery to classify land cover using the Scene Classification Layer (SCL) as ground truth. It leverages transfer learning with a DeepLabv3-ResNet50 backbone, starting with training only the classifier head and later partially fine-tuning deeper layers.

---

## Project Overview

- **Objective**: 
  To classify land cover classes from Sentinel-2 true color images using the SCL mask as reference.

- **Workflow**:
  1. Download Sentinel-2 images from Copernicus Open Access Hub (True Color at 20m) and SCL masks.
  2. Split the large images into patches (512x512) for easier processing and training.
  3. Visualize the True Color images, SCL masks, and overlay.
  4. Train a DeepLabv3-ResNet50 model:
     - Initially freeze the backbone and train only the classifier head.
     - Then unlock only the last ResNet block (`layer4`) for partial fine-tuning.
  5. Plot learning curves and evaluate pixel accuracy & IoU.

---

## Dataset

- **Input**: Sentinel-2 True Color images (`B4`, `B3`, `B2`) at 20m resolution.
- **Labels**: Sentinel-2 SCL masks (Scene Classification Layer), which classifies pixels into 11 categories (clouds, vegetation, water, etc).

- **Preprocessing**:
  - Normalized images by dividing by 255.
  - Converted masks to long tensors (0 to 10).

---

## Model Architecture

- **Backbone**: Pretrained DeepLabv3 with ResNet50.
- **Custom head**: Modified to output 11 classes instead of original 21.

Training strategy:
- **Epochs 0-10**: Train only classifier head (backbone frozen).
- **Epochs 10+**: Unlock only `layer4` of ResNet for partial fine-tuning.

This approach balances leveraging pretrained features and adapting to the specific dataset.

---

## Results

### Example visualization of the dataset
![Dataset Visualization](/figures/example_1.png)

- Left: True Color Image
- Center: SCL mask
- Right: Overlay

---

### Fine-tuning of only the head

### Learning Curves & Validation Metrics
![Training Curves](/figures/curves_1.png)

- Left: Training and validation loss.
- Right: Pixel accuracy and IoU on validation set.

The model reached a pixel accuracy of ~90% and stable IoU after 40 epochs.

### Partial Fine-tuning of the head and the layer 4

![Training Curves](/figures/Curve_3.png)

The accuracy during the partial fine-tuning is even higher as expected.

---

### Results

Here I show an example of a really hard mask positioned in Malaysia:
![Example](/figures/Figure_5.png)

- Left: The image given as input to the model
- Center: The actual mask of the image
- Right: The predicted mask from the model

As we can see the original mask is hugely more granular than the predicted one.

## Future updates

To improve the results, I'd like to do a full training of the architecture.

## ✍️ Author

- Riccardo Marasca


