# VisionGuard 🛡️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Alibubere/visionguard/graphs/commit-activity)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/Alibubere/visionguard)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Computer%20Vision-orange.svg)](https://github.com/Alibubere/visionguard)
[![Model](https://img.shields.io/badge/Model-Mask%20R--CNN-purple.svg)](https://arxiv.org/abs/1703.06870)
[![Dataset](https://img.shields.io/badge/Dataset-COCO%20Format-yellow.svg)](https://cocodataset.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/Alibubere/visionguard)

> A robust computer vision pipeline for object detection and instance segmentation using Mask R-CNN on custom datasets.

## 🚀 Features

- **Advanced Object Detection**: Powered by Mask R-CNN with ResNet-50 FPN backbone
- **Instance Segmentation**: Precise pixel-level object segmentation
- **COCO Dataset Integration**: Seamless handling of COCO format annotations
- **Flexible Configuration**: YAML-based configuration management
- **Comprehensive Logging**: Detailed training and evaluation logs
- **Checkpoint Management**: Model saving and resuming capabilities
- **Multi-Class Support**: Configurable for up to 15 object classes
- **GPU Acceleration**: CUDA support for faster training

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset Format](#dataset-format)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Alibubere/visionguard.git
   cd visionguard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install pyyaml opencv-python pillow matplotlib numpy
   ```

## ⚡ Quick Start

1. **Prepare your dataset** in COCO format
2. **Update configuration**:
   ```bash
   cp configs/config.yaml configs/my_config.yaml
   # Edit configs/my_config.yaml with your paths
   ```
3. **Run training**:
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
visionguard/
├── 📁 checkpoints/          # Model checkpoints
├── 📁 configs/              # Configuration files
│   └── config.yaml          # Main configuration
├── 📁 data/                 # Dataset directory
│   ├── 📁 processed/        # Processed annotations
│   └── 📁 raw/              # Raw dataset files
├── 📁 logs/                 # Training logs
├── 📁 src/                  # Source code
│   ├── 📁 work_data/        # Data processing modules
│   │   ├── dataloader.py    # Data loading utilities
│   │   ├── dataset.py       # Dataset classes
│   │   └── merge_coco.py    # COCO annotation merger
│   ├── model.py             # Model definitions
│   └── train.py             # Training utilities
├── main.py                  # Main execution script
└── README.md               # This file
```

## ⚙️ Configuration

The `configs/config.yaml` file contains all training parameters:

```yaml
data:
  data_dir: data/raw/vision
  output_dir: data/processed/annotations/
  # ... other data paths

training:
  num_epochs: 20
  batch_size: 2
  lr: 0.005
  weight_decay: 0.0005

model:
  num_classes: 15
  freeze_backbone: false
```

## 🎯 Usage

### Training a Model

```bash
python main.py
```

### Custom Configuration

```bash
# Modify configs/config.yaml or create a new config file
python main.py --config configs/custom_config.yaml
```

## 🏗️ Model Architecture

- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Detection Head**: Faster R-CNN for object detection
- **Segmentation Head**: Mask R-CNN for instance segmentation
- **Classes**: Configurable (default: 15 classes)

### Supported Object Classes

The model supports detection of various objects including:
- Cable, Capacitor, Wood, and 12 other configurable classes

## 📊 Dataset Format

VisionGuard expects datasets in COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [...],
      "area": 1234
    }
  ],
  "categories": [...]
}
```

## 🏋️ Training

### Training Process

1. **Data Merging**: Combines multiple COCO annotation files
2. **Dataset Loading**: Creates PyTorch datasets and dataloaders
3. **Model Initialization**: Sets up Mask R-CNN with custom heads
4. **Training Loop**: Trains for specified epochs with logging
5. **Validation**: Evaluates model performance each epoch

### Training Features

- ✅ Automatic checkpoint saving
- ✅ Learning rate scheduling
- ✅ Comprehensive logging
- ✅ GPU acceleration
- ✅ Resume from checkpoint

## 📈 Evaluation

The model outputs detailed metrics including:
- Training loss per epoch
- Validation loss per epoch
- Individual loss components (classification, bbox regression, mask)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ali Muin**
- 🌐 GitHub: [Alibubere](https://github.com/Alibubere)
- 📧 Email: alibubere989@gmail.com
- 💼 LinkedIn: [Mohammad Ali Bubere](https://www.linkedin.com/in/mohammad-ali-bubere-a6b830384/)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Ali Bubere](https://github.com/Alibubere)

</div>

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [torchvision](https://pytorch.org/vision/) for computer vision utilities
- [COCO Dataset](https://cocodataset.org/) for the annotation format standard
- [Mask R-CNN](https://arxiv.org/abs/1703.06870) paper by He et al.

---

*Last updated: 4-12-2025*