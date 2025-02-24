#!/usr/bin/env python
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from model import UNetDeblurImproved

# --------------------------------------------
# Improved U-Net–Style Architecture for Deblurring
# (Same as in train.py)
# --------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py input_blurred.png output_deblurred.png")
        exit(1)
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDeblurImproved().to(device)
    model.load_state_dict(torch.load("deblur_net.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((36, 1100)),
        transforms.ToTensor(),
    ])

    input_img = Image.open(input_image_path).convert("L")
    input_tensor = transform(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
    output_img = transforms.ToPILImage()(output_tensor)
    output_img.save(output_image_path)
    print(f"Saved deblurred image to {output_image_path}")

if __name__ == "__main__":
    main()