# Chest X-ray Lung Segmentation

A deep learning project for automated lung segmentation in chest X-ray images using PyTorch and U-Net architecture.

## Overview

This project implements a complete pipeline for lung segmentation in chest X-ray images, including data visualization, preprocessing, model training, and evaluation. The model is trained on a combined dataset of 3,211 chest X-ray images with corresponding lung segmentation masks.

## Dataset

The dataset combines images from three sources:
- **Darwin Dataset**: 2,507 images (including normal, bacterial, and viral pneumonia cases)
- **Montgomery Dataset**: 138 images
- **Shenzhen Dataset**: 566 images

**Total**: 3,211 image-mask pairs in PNG format

### Dataset Structure
```
Dataset/ChestXray/
├── CXR_Combined/
│   ├── images/          # 3,211 chest X-ray images
│   └── masks/           # 3,211 corresponding lung masks
├── CXR_RadioOpaque/     # Images with pneumonia (radio-opaque)
├── CXR_RadioOpaque_Mask/
├── CXR_RadioLucent/     # Normal images (radio-lucent)
├── CXR_RadioLucent_Mask/
└── CXR_Selected-Image-Dataset_Log.csv
```

### Image Naming Convention
- `DARCXR_*.png` - Darwin dataset images
- `CHNCXR_*_*.png` - Shenzhen dataset images
- `MCUCXR_*_*.png` - Montgomery dataset images

## Project Structure

```
ChestXraySegmentation/
├── visualization.ipynb      # Data analysis and visualization
├── training.ipynb          # Model training and evaluation
├── README.md              # This file
├── train_split.csv        # Training set filenames (70%)
├── val_split.csv          # Validation set filenames (15%)
├── test_split.csv         # Test set filenames (15%)
├── models/                # Saved model checkpoints
│   ├── best_model.pth
│   ├── final_model.pth
│   └── checkpoint_epoch_*.pth
├── results/               # Training results and metrics
│   ├── training_history.json
│   ├── test_metrics.json
│   └── test_detailed_metrics.csv
└── plots/                 # Generated visualizations
    ├── training_curves.png
    ├── test_predictions.png
    └── metrics_distribution.png
```

## Notebooks

### 1. visualization.ipynb
**Purpose**: Analyze and prepare the dataset for training

**Features**:
- Dataset statistics and distribution analysis
- Image dimension analysis
- Mask coverage percentage calculations
- Train/validation/test split creation (70/15/15)
- Multiple visualization types:
  - Dataset source distribution
  - Image dimension distributions
  - Sample images with mask overlays
  - Mask coverage analysis
- Export train/val/test splits to CSV files

**Output**:
- `train_split.csv` - 2,247 images
- `val_split.csv` - 482 images
- `test_split.csv` - 482 images

### 2. training.ipynb
**Purpose**: Train U-Net model for lung segmentation

**Features**:
- Custom PyTorch Dataset with data augmentation
- U-Net architecture implementation (~31M parameters)
- Combined loss function (BCE + Dice Loss)
- GPU acceleration with CUDA support
- Training with early stopping
- Comprehensive metrics (IoU, Dice, Precision, Recall)
- Model checkpointing (best, final, periodic)
- Training visualization and monitoring
- Test set evaluation with per-image metrics

**Model Architecture**:
- Input: 256×256 grayscale images
- Encoder: 4 downsampling blocks with skip connections
- Bottleneck: 512 channels
- Decoder: 4 upsampling blocks
- Output: Binary segmentation mask

## Requirements

### Python Packages
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python>=4.8.0
Pillow>=10.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

### Hardware
- GPU with CUDA support (recommended for training)
- At least 8GB RAM
- ~5GB free disk space for models and results

## Usage

### Step 1: Data Visualization and Preparation
```bash
# Open and run visualization.ipynb
# This will:
# - Analyze the dataset
# - Create train/val/test splits
# - Generate visualizations
# - Export split CSV files
```

### Step 2: Model Training
```bash
# Open and run training.ipynb
# This will:
# - Load the dataset splits
# - Train the U-Net model on GPU
# - Save model checkpoints
# - Generate training curves
# - Evaluate on test set
```

### Training Configuration
- **Batch Size**: 16
- **Epochs**: 50 (with early stopping)
- **Learning Rate**: 1e-4
- **Optimizer**: Adam (weight decay: 1e-5)
- **Loss Function**: Combined BCE + Dice Loss (alpha=0.5)
- **Image Size**: 256×256 pixels
- **Data Augmentation**: Horizontal flips (50% probability)

## Results

The model generates the following outputs:

### Saved Models
- `best_model.pth` - Model with best validation loss
- `final_model.pth` - Model after final epoch
- `checkpoint_epoch_*.pth` - Periodic checkpoints

### Metrics
- **IoU (Intersection over Union)**: Overlap accuracy
- **Dice Coefficient**: Segmentation similarity
- **Precision**: Correct positive predictions
- **Recall**: Coverage of actual positives

### Visualizations
- Training/validation loss curves
- Metric curves (IoU, Dice)
- Sample predictions with ground truth
- Metric distribution histograms

## Model Performance

The U-Net model is trained to segment lungs in chest X-ray images with high accuracy. Performance metrics are saved in:
- `results/training_history.json` - Training and validation metrics per epoch
- `results/test_metrics.json` - Final test set performance
- `results/test_detailed_metrics.csv` - Per-image metrics for all test images

## Dataset Attribution

Dataset source: [Mendeley Data](https://data.mendeley.com/datasets/8gf9vpkhgy/1)

Original datasets:
1. **Darwin Dataset**: 6,106 images (2,507 selected)
2. **Montgomery Dataset**: 139 images (138 used)
3. **Shenzhen Dataset**: 566 images (100% used)

Dataset preparation: See `Dataset/ChestXray/prepare_image_dataset.ipynb`

## Author

**Mrunal Shah**
- Email: mrunalnshah@protonmail.com
- GitHub: [mrunalnshah](https://github.com/mrunalnshah)
- LinkedIn: [mrunalnshah](https://www.linkedin.com/in/mrunalnshah/)

## License

This project is for educational and research purposes. Please refer to the original dataset licenses for usage restrictions.

## Acknowledgments

- Dataset providers: Darwin, Montgomery, and Shenzhen hospitals
- U-Net architecture: Ronneberger et al. (2015)
- PyTorch framework and community

## Notes

- The dataset is balanced with radio-opaque (pneumonia) and radio-lucent (normal) images
- Images are automatically resized to 256×256 during training
- GPU is automatically detected and used when available
- Training progress is displayed with progress bars
- All visualizations are automatically saved to the `plots/` directory

## Troubleshooting

**Out of Memory Error**: Reduce batch size in training.ipynb
**CUDA Not Available**: Training will automatically fall back to CPU
**Missing Splits**: Run visualization.ipynb first to generate CSV files
**Path Errors**: Ensure the Dataset/ChestXray/ directory is accessible

---

For questions or issues, please open an issue on the GitHub repository.
