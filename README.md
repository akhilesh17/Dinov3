# DINOv3 Object Similarity Detection

A powerful image analysis tool using DINOv3 (Vision Transformer-based self-supervised learning model) to detect and localize objects across images through patch-level feature matching and cosine similarity.

## Overview

This project leverages DINOv3 ViT-S/16 to:
- Extract dense patch-level features from images
- Create embeddings from reference objects
- Compute similarity maps to localize objects in other images
- Generate visualization overlays with similarity heatmaps

The approach is ideal for object detection, visual search, and similarity-based image analysis without requiring labeled training data.

## Features

✨ **Key Capabilities:**
- **Patch-based Feature Extraction**: Extracts features from 14×14 patch tokens using DINOv3
- **Object Embedding**: Creates robust object representations by averaging top-K salient patches
- **Similarity Mapping**: Computes cosine similarity between patches and object embeddings
- **Visualization**: Generates heatmap overlays showing object location confidence
- **GPU Support**: Automatic CUDA acceleration when available

## Project Structure

```
.
├── compare_image_dinov3_chatgpt.ipynb          # Main analysis notebook
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth # Pre-trained DINOv3 model
└── README.md                                    # This file
```

## Requirements

- Python 3.8+
- PyTorch
- timm (PyTorch Image Models)
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pillow (PIL)
- torchvision

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install torch torchvision timm opencv-python numpy matplotlib pillow
```

3. Ensure the model checkpoint `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` is in the project directory

## Usage

### Basic Workflow

```python
from notebook # See compare_image_dinov3_chatgpt.ipynb

# 1. Load the DINOv3 model
model = load_dino_model(model_path, device)

# 2. Load images
img1_pil, img1_tensor = load_image(image1_path)  # Query image
img2_pil, img2_tensor = load_image(image2_path)  # Reference object

# 3. Extract patch features
features1 = extract_patch_features(model, img1_tensor)
features2 = extract_patch_features(model, img2_tensor)

# 4. Create object embedding from reference
object_embedding = create_object_embedding(features2, k=10)

# 5. Compute similarity map
sim_map = compute_similarity_map(features1, object_embedding)

# 6. Visualize results
visualize_overlay_with_colorbar(img1_pil, sim_map, alpha=0.5)
```

### Core Functions

#### `load_dino_model(model_path, device)`
Loads the DINOv3 ViT-S/16 model from checkpoint.

#### `load_image(path)`
Preprocesses an image: resizes to 224×224, normalizes, and converts to tensor.

#### `extract_patch_features(model, image_tensor)`
Extracts patch token features from the vision transformer (removes CLS token, returns 196 patch features).

#### `create_object_embedding(features, k=10)`
Creates an object embedding by:
1. Normalizing patch features
2. Selecting top-K patches by feature norm
3. Averaging to create a unified embedding

#### `compute_similarity_map(features1, object_embedding)`
Computes cosine similarity between all patches and the object embedding, reshapes to 14×14 grid.

#### `visualize_overlay_with_colorbar(img, sim_map, alpha=0.5)`
Generates a heatmap overlay showing similarity scores across the image.

## How It Works

1. **Feature Extraction**: DINOv3 encodes images into 196 patch tokens (14×14 grid) with dimension 384
2. **Object Representation**: Top-K salient patches from the reference object are averaged
3. **Similarity Search**: Cosine similarity between all patches and object embedding creates a spatial map
4. **Localization**: The similarity map indicates where the object appears in the query image

## Example Use Cases

- **Visual Search**: Find similar objects across images
- **Object Localization**: Identify where objects of interest appear
- **Image Comparison**: Analyze patch-level similarity between images
- **Anomaly Detection**: Detect objects different from a reference

## Model Information

- **Architecture**: Vision Transformer (ViT-S/16)
- **Training**: Self-supervised (DINO methodology)
- **Input Size**: 224×224 pixels
- **Feature Dimension**: 384
- **Patch Size**: 16×16
- **Patches**: 14×14 = 196 tokens

## Notes

- The model runs faster on GPU; CPU mode is supported but slower
- Image normalization uses ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Similarity threshold can be adjusted for bounding box extraction
- The top-K parameter (default k=10) affects object embedding robustness

## References

- DINOv3: [Facebook Research](https://github.com/facebookresearch/dino)
- Vision Transformers: [ViT Paper](https://arxiv.org/abs/2010.11929)
- TIMM Library: [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)

## License

This project uses pre-trained models. Ensure compliance with their respective licenses.
