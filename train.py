#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from model import UNetDeblurImproved
import logging
from datetime import datetime

# --------------------------------------------
# 1. Custom Dataset for Paired Deblurring Data
# --------------------------------------------
class DeblurDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, transform=None):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        # Ensure images are sorted so that pairs match
        self.blur_images = sorted(os.listdir(blur_dir))
        self.sharp_images = sorted(os.listdir(sharp_dir))
        self.transform = transform

    def __len__(self):
        return len(self.blur_images)

    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_dir, self.blur_images[idx])
        sharp_path = os.path.join(self.sharp_dir, self.sharp_images[idx])
        blur_img = Image.open(blur_path).convert('L')   # grayscale
        sharp_img = Image.open(sharp_path).convert('L')   # grayscale

        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
        return blur_img, sharp_img

# --------------------------------------------
# 3. Training Loop
# --------------------------------------------
def setup_logger():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    logger = logging.getLogger(__name__)
    best_val_loss = float('inf')
    
    logger.info(f"Starting training on device: {device}")
    logger.info(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for i, (blur, sharp) in enumerate(train_loader):
            blur = blur.to(device)
            sharp = sharp.to(device)
            optimizer.zero_grad()
            output = model(blur)
            loss = criterion(output, sharp)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * blur.size(0)
        train_loss = running_train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for blur, sharp in val_loader:
                blur = blur.to(device)
                sharp = sharp.to(device)
                output = model(blur)
                loss = criterion(output, sharp)
                running_val_loss += loss.item() * blur.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "deblur_net.pth")
            logger.info(f"New best model saved! (Val Loss: {val_loss:.4f})")

# --------------------------------------------
# 4. Main: Data Preparation and Training
# --------------------------------------------
def main():
    logger = setup_logger()
    
    parser = argparse.ArgumentParser(description="Train DeepTextDeblur Model")
    parser.add_argument("--restart", action="store_true", help="Restart training from scratch (ignore existing checkpoint)")
    args = parser.parse_args()

    # Directories for blurred and sharp images
    blur_dir = 'data/blur'
    sharp_dir = 'data/sharp'

    # Transformation: resize to 36 x 1100, random horizontal flip, and tensor conversion
    transform = transforms.Compose([
        transforms.Resize((36, 1100)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = DeblurDataset(blur_dir, sharp_dir, transform=transform)
    print("Total samples:", len(dataset))

    # Split dataset: 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create DataLoaders
    num_workers = min(8, os.cpu_count())  # Automatically use appropriate number of workers
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss, optimizer
    model = UNetDeblurImproved().to(device)
    # If a checkpoint exists and --restart is NOT specified, load the checkpoint
    if not args.restart and os.path.exists("deblur_net.pth"):
        print("Found existing checkpoint, continuing training from saved weights...")
        model.load_state_dict(torch.load("deblur_net.pth", map_location=device))
    else:
        print("Starting training from scratch.")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100  # Adjust as needed

    # Log training configuration
    logger.info("Training configuration:")
    logger.info(f"Batch size: 8")
    logger.info(f"Learning rate: 1e-3")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Using device: {device}")
    
    if not args.restart and os.path.exists("deblur_net.pth"):
        logger.info("Loading existing checkpoint...")
    else:
        logger.info("Starting training from scratch")

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)

if __name__ == '__main__':
    main()