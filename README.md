# DeepTextDeblur

DeepTextDeblur is a deep learning project for deblurring images containing text.

## Directory Structure
```
DeepTextDeblur/
├── data/
│   └── Blur/                 # Folder with blurred text images
│   └── Sharp/                # Folder with corresponding sharp text images
├── train.py                  # Training script for the deblurring network
├── run.py                    # Inference script for deblurring a single image
├── generate.py               # [Optional: Additional generation utility]
├── distribution.py           # [Optional: Additional distribution utility]
├── run.sh                    # Shell script to launch training
└── README.md                 # This file
```

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/DeepTextDeblur.git
cd DeepTextDeblur
```

Set up a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

## Image Generation

Generate sharp text images:
```
python generate.py
```

After generating sharp images using `generate.py`, the images need to be blurred to create the blurred dataset. This project uses GIMP's Gaussian blur filter for consistent and high-quality blurring.

### Prerequisites

1. Install GIMP:
   - **Ubuntu/Debian**: `sudo apt-get install gimp`
   - **macOS**: `brew install --cask gimp` or download from [GIMP website](https://www.gimp.org/downloads/)
   - **Windows**: Download and install from [GIMP website](https://www.gimp.org/downloads/)

2. Ensure GNU Parallel is installed:
   - **Ubuntu/Debian**: `sudo apt-get install parallel`
   - **macOS**: `brew install parallel`
   - **Windows**: Install via WSL or download from [GNU Parallel](https://www.gnu.org/software/parallel/)

### Running the Blur Script

The `blur_parallel.sh` script processes all sharp images in parallel using GIMP's batch processing capabilities:
```
./blur_parallel.sh
```

This will create a `data/blur` directory with the blurred images.

## Training
- Organize your dataset into the following structure:
  - data/blur: Contains blurred text images.
  - data/sharp: Contains corresponding sharp text images. 
- Launch the training process by running:
```
./run.sh
```
This will execute train.py, which splits your dataset (80% training, 20% validation), trains the improved U-Net model, and saves the best model weights as deblur_net.pth.

## Inference

After training, deblur a new image using the inference script:
```
python run.py input_blurred.png output_deblurred.png
```

Replace input_blurred.png with the filename of your blurred image and output_deblurred.png with the desired output filename. The script loads the saved model and processes the image to produce a deblurred result.

## Architecture

The model uses a modified U-Net architecture optimized for text deblurring:

![Neural Network Structure](static/img/SHCStrc_1.png)

The network features:
- Multiple skip connections for better feature preservation
- Deep encoder-decoder structure
- Specialized convolution blocks for text detail recovery

## Acknowledgments

This project is inspired by the [DeepDeblur](https://github.com/SeungjunNah/DeepDeblur) repository and related research on image deblurring. Special thanks to the deep learning community for open-source contributions that made this project possible.