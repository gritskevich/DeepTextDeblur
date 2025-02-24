import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDeblurImproved(nn.Module):
    def __init__(self):
        super(UNetDeblurImproved, self).__init__()
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = self.conv_block(256, 512)
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
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
        _, _, h_enc, w_enc = enc_feature.size()
        _, _, h_target, w_target = target.size()
        diff_h = h_enc - h_target
        diff_w = w_enc - w_target
        return enc_feature[:, :, diff_h // 2: h_enc - (diff_h - diff_h // 2),
                           diff_w // 2: w_enc - (diff_w - diff_w // 2)]

    def forward(self, x):
        input_shape = x.size()
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.upconv4(b)
        if d4.size()[2:] != e4.size()[2:]:
            e4 = self.center_crop(e4, d4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        if d3.size()[2:] != e3.size()[2:]:
            e3 = self.center_crop(e3, d3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        if d2.size()[2:] != e2.size()[2:]:
            e2 = self.center_crop(e2, d2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        if d1.size()[2:] != e1.size()[2:]:
            e1 = self.center_crop(e1, d1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.conv_final(d1)
        
        target_h, target_w = input_shape[2], input_shape[3]
        out_h, out_w = out.size(2), out.size(3)
        diff_h = target_h - out_h
        diff_w = target_w - out_w
        if diff_h > 0 or diff_w > 0:
            out = F.pad(out, (diff_w // 2, diff_w - diff_w // 2,
                              diff_h // 2, diff_h - diff_h // 2))
        elif diff_h < 0 or diff_w < 0:
            out = self.center_crop(out, x)
        return out 