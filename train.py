#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

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
# 2. Improved U-Net–Style Architecture for Deblurring
#    (Deeper with 4 encoder/decoder levels)
# --------------------------------------------
class UNetDeblurImproved(nn.Module):
    def __init__(self):
        super(UNetDeblurImproved, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(1, 32)    # [B,32,36,1100]
        self.enc2 = self.conv_block(32, 64)     # After pool: [B,64,18,550]
        self.enc3 = self.conv_block(64, 128)    # After pool: [B,128,9,275]
        self.enc4 = self.conv_block(128, 256)   # After pool: [B,256,4,137]
        self.pool = nn.MaxPool2d(2)             # kernel_size=2, stride=2

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)  # After pool: [B,512,2,68]

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # [B,256,4,136]
        self.dec4 = self.conv_block(512, 256)  # concat with enc4 => [B,512,4,136] -> [B,256,4,136]

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # [B,128,8,272]
        self.dec3 = self.conv_block(256, 128)  # concat with enc3 => [B,256,8,272] -> [B,128,8,272]

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # [B,64,16,544]
        self.dec2 = self.conv_block(128, 64)   # concat with enc2 => [B,128,16,544] -> [B,64,16,544]

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # [B,32,32,1088]
        self.dec1 = self.conv_block(64, 32)    # concat with enc1 => [B,64,32,1088] -> [B,32,32,1088]

        # Final 1x1 conv to get back to one channel
        self.conv_final = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def center_crop(self, enc_feature, target):
        """
        Center-crop enc_feature to match the spatial size of target.
        """
        _, _, h_enc, w_enc = enc_feature.size()
        _, _, h_target, w_target = target.size()
        diff_h = h_enc - h_target
        diff_w = w_enc - w_target
        return enc_feature[:, :, diff_h // 2: h_enc - (diff_h - diff_h // 2),
               diff_w // 2: w_enc - (diff_w - diff_w // 2)]

    def forward(self, x):
        input_shape = x.size()  # Save original size

        # Encoder
        e1 = self.enc1(x)                   # [B,32,36,1100]
        e2 = self.enc2(self.pool(e1))         # [B,64,18,550]
        e3 = self.enc3(self.pool(e2))         # [B,128,9,275]
        e4 = self.enc4(self.pool(e3))         # [B,256,4,137]

        b = self.bottleneck(self.pool(e4))    # [B,512,2,68]

        # Decoder
        d4 = self.upconv4(b)                  # [B,256,4,136]
        if d4.size()[2:] != e4.size()[2:]:
            e4 = self.center_crop(e4, d4)
        d4 = torch.cat([d4, e4], dim=1)         # [B,512,4,136]
        d4 = self.dec4(d4)                    # [B,256,4,136]

        d3 = self.upconv3(d4)                 # [B,128,8,272]
        if d3.size()[2:] != e3.size()[2:]:
            e3 = self.center_crop(e3, d3)
        d3 = torch.cat([d3, e3], dim=1)         # [B,256,8,272]
        d3 = self.dec3(d3)                    # [B,128,8,272]

        d2 = self.upconv2(d3)                 # [B,64,16,544]
        if d2.size()[2:] != e2.size()[2:]:
            e2 = self.center_crop(e2, d2)
        d2 = torch.cat([d2, e2], dim=1)         # [B,128,16,544]
        d2 = self.dec2(d2)                    # [B,64,16,544]

        d1 = self.upconv1(d2)                 # [B,32,32,1088]
        if d1.size()[2:] != e1.size()[2:]:
            e1 = self.center_crop(e1, d1)
        d1 = torch.cat([d1, e1], dim=1)         # [B,64,32,1088]
        d1 = self.dec1(d1)                    # [B,32,32,1088]

        out = self.conv_final(d1)             # [B,1,32,1088]

        # Final adjustment: pad/crop to match input size (36, 1100)
        target_h, target_w = input_shape[2], input_shape[3]  # (36,1100)
        out_h, out_w = out.size(2), out.size(3)               # (32,1088)
        diff_h = target_h - out_h  # Expected: 4
        diff_w = target_w - out_w  # Expected: 12
        if diff_h > 0 or diff_w > 0:
            out = F.pad(out, (diff_w // 2, diff_w - diff_w // 2,
                              diff_h // 2, diff_h - diff_h // 2))
        elif diff_h < 0 or diff_w < 0:
            out = self.center_crop(out, x)
        return out

# --------------------------------------------
# 3. Training Loop
# --------------------------------------------
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for blur, sharp in train_loader:
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "deblur_net.pth")
            print("Saved best model!")

# --------------------------------------------
# 4. Main: Data Preparation and Training
# --------------------------------------------
def main():
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
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, loss function, and optimizer
    model = UNetDeblurImproved().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100  # Adjust as needed

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)

if __name__ == '__main__':
    main()