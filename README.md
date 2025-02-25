# Pixels and Secrets: Unmasking Blurred Text with Neural Networks

![Enhance Meme](static/img/enhance.png)

## Introduction

Have you ever looked at a blurred screenshot in documentation and thought, "I wonder what that actually says?" Most people would shrug and move on, but if you're reading this, you're probably not most people. Let's embark on a journey to recover text that someone deliberately obscured.

This article describes my adventure with the [DeepTextDeblur](https://github.com/gritskevich/DeepTextDeblur) project, where I tackle the challenge of recovering blurred text from documentation using neural networks. It's a tale of curiosity, machine learning, and perhaps a gentle reminder about security practices.

## The Mystery Begins

It all started when I was reading through some administration documentation and came across a peculiar section about licenses. The guide helpfully explained how to add a license key, but the accompanying screenshot had the actual license key blurred out.

![License](static/img/license.png)

This got me thinking: is blurring text actually a secure way to redact information? Or is it just security theater?

## The Quest

Instead of just wondering, I decided to find out. Enter DeepTextDeblur, a project inspired by [DeepDeblur](https://github.com/meijianhan/DeepDeblur) but specifically focused on text recovery. 

The core idea is straightforward but powerful: train a neural network to reverse the blurring process by learning the relationship between blurred text and its original form.

## The Approach

### 1. Data Generation

First, I needed training data - lots of it. I created a script that:
- Generates random license keys that match the format I observed
- Renders them as images
- Applies various blurring techniques to create pairs of clear/blurred images

![Generated Sharp](static/img/sharp_0000.png)
![Generated Blur](static/img/blur_0000.png)

### 2. The Neural Network

The architecture is a modified U-Net with attention mechanisms designed specifically for text recovery:

```python
# The heart of our deblurring network
# (Not the actual implementation, just illustrating the concept)
def build_deblur_model():
    inputs = Input(shape=(None, None, 3))
    
    # Encoder path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    # ... more layers ...
    
    # Decoder with skip connections
    up1 = concatenate([UpSampling2D()(conv4), conv3])
    # ... more layers ...
    
    outputs = Conv2D(3, 3, activation='sigmoid', padding='same')(up2)
    
    return Model(inputs, outputs)
```
![NN](static/img/nn.png)


This model learns to map blurred images back to their sharp counterparts. The magic happens in those skip connections, which help preserve the fine details that make text recognizable.

### 3. Training Adventures

Training was... well, let's just say my GPU fan got a workout. Each epoch took about 10-15 minutes, and I watched as the model gradually improved:

```
Epoch 1/100
Loss: 0.4827 - Accuracy: 0.2145
Epoch 2/100
Loss: 0.3912 - Accuracy: 0.3567
...
Epoch 47/100
Loss: 0.0823 - Accuracy: 0.9134
...
Epoch 100/100
Loss: 0.0312 - Accuracy: 0.9678
```

I found myself checking in every few epochs, watching as gibberish slowly transformed into recognizable characters. It felt like developing a photo in a darkroom - the image gradually emerging from nothing.

## The Reveal

After training completed, it was time for the moment of truth. I fed the model the blurred license key from the documentation and...

![License Blur](static/img/blur_license.png)
![License Sharp](static/img/sharp_license.png)

Success! The model recovered text that was clearly a license key format with remarkable accuracy. While not perfect, it was more than enough to understand the obscured information.

## Beyond Deblurring: A Deeper Dive

While recovering the text was fun, I wondered if there was a more direct approach. Using tools like a Python bytecode decompiler and an HTTPS proxy, it became clear that the real solution wasn't about breaking the blur - it was about understanding the underlying system.

Without going into specifics that might compromise any systems, I discovered that with the right understanding of the communication protocol, one could theoretically:
- Intercept license verification requests
- Modify DNS to point to a custom server
- Create responses that would satisfy the verification process
- Replace compiled `.pyc` files with modified versions that bypass license validation entirely

## Conclusion

This journey wasn't just about breaking a blur effect - it was a reminder that in the digital age, we need to be thoughtful about how we protect sensitive information. If you're including screenshots in your documentation, remember that blurring text is about as secure as hiding a key under your doormat.


## Technical Implementation

For those interested in the technical details or who want to try DeepTextDeblur themselves, this section provides the necessary information to get started.

### Project Repository

The DeepTextDeblur project is available on GitHub: [github.com/gritskevich/DeepTextDeblur](https://github.com/gritskevich/DeepTextDeblur)

### Directory Structure

```
DeepTextDeblur/
├── data/
│   ├── blur/                 # Folder with blurred text images
│   └── sharp/                # Folder with corresponding sharp text images
├── train.py                  # Training script for the deblurring network
├── run.py                    # Inference script for deblurring a single image
├── generate.py               # Text image generation utility
├── blur_parallel.sh          # Parallel blurring utility using GIMP
└── README.md                 # Project documentation
```

### Installation

Getting started with DeepTextDeblur is straightforward:

```bash
# Clone the repository
git clone https://github.com/gritskevich/DeepTextDeblur.git
cd DeepTextDeblur

# Set up a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Creation

The power of this project comes from its ability to generate synthetic training data that closely resembles real-world scenarios.

#### 1. Generate Sharp Text Images

The first step is to create clear text images that will serve as the ground truth:

```bash
python generate.py
```

This script randomly generates text samples that mimic license keys and other structured text formats, rendering them as high-quality images.

#### 2. Create Blurred Counterparts

After generating the sharp images, we need to create realistic blur effects:

```bash
./blur_parallel.sh
```

This script utilizes GIMP's professional-grade Gaussian blur filter and GNU Parallel to efficiently process all images.

### Prerequisites for Blurring

- **GIMP**: Professional image manipulation program
  - Ubuntu/Debian: `sudo apt-get install gimp`
  - macOS: `brew install --cask gimp` 
  - Windows: Download from [GIMP website](https://www.gimp.org/downloads/)

- **GNU Parallel**: Tool for parallel processing
  - Ubuntu/Debian: `sudo apt-get install parallel`
  - macOS: `brew install parallel`
  - Windows: Available via WSL

### Training the Model

Once your dataset is prepared, training the neural network is a single command:

```bash
python train.py
```

This process:
1. Splits your dataset (80% training, 20% validation)
2. Initializes the U-Net architecture with attention mechanisms
3. Trains the model, saving checkpoints along the way
4. Outputs the final model as `deblur_net.pth`

Each training run typically takes several hours on a modern GPU, with noticeable improvements in quality after about 50 epochs.

### Running Inference

After training, you can deblur new images using:

```bash
python run.py input_blurred.png output_deblurred.png
```

Simply replace `input_blurred.png` with your blurred image and `output_deblurred.png` with your desired output filename.

### Neural Network Architecture

The heart of DeepTextDeblur is a modified U-Net architecture optimized specifically for text recovery:

Key architectural features include:
- **Skip connections** that preserve spatial information across the network
- **Deep encoder-decoder structure** for progressive feature extraction and reconstruction
- **Attention mechanisms** that help focus on text regions
- **Specialized convolution blocks** designed for recovering fine text details

## Acknowledgments

This project builds upon the foundational work of [DeepDeblur](https://github.com/meijianhan/DeepDeblur) and related research in the field of image deblurring. The modifications made focus specifically on optimizing for text recovery rather than general image deblurring.

---

*Disclaimer: This project was created for educational purposes. The techniques described should not be used to access information you don't have permission to view. Always respect privacy and security boundaries.*