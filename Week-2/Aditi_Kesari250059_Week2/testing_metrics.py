import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

# 1. Redefine the U-Net Architecture (Required to load weights)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        bn = self.bottleneck(self.pool4(d4))
        u1 = self.conv_up1(torch.cat([self.up1(bn), d4], dim=1))
        u2 = self.conv_up2(torch.cat([self.up2(u1), d3], dim=1))
        u3 = self.conv_up3(torch.cat([self.up3(u2), d2], dim=1))
        u4 = self.conv_up4(torch.cat([self.up4(u3), d1], dim=1))
        return self.outc(u4)

# 2. Helper Functions
def rle_decode(mask_rle, shape):
    """Decodes Kaggle RLE string into a 2D binary mask."""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1.0
    return img.reshape(shape).T

def calculate_metrics(pred, target):
    """Calculates IoU and Dice Coefficient."""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = intersection / (union + 1e-6)
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
    return iou, dice

# 3. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet().to(device)
model.load_state_dict(torch.load('unet_model.pth', map_location=device))
model.eval()

test_dir = 'data-science-bowl-2018/stage1_test'
solution_df = pd.read_csv('data-science-bowl-2018/stage1_solution.csv')

image_ids = next(os.walk(test_dir))[1]
ious = []
dices = []

print("Running inference and calculating metrics on stage1_test...")

# 4. Evaluation Loop
for img_id in image_ids:
    # Read original image
    img_path = os.path.join(test_dir, img_id, 'images', img_id + '.png')
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    original_shape = image.shape[:2]
    
    # Preprocess for model
    image_resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (128, 128)) / 255.0
    image_tensor = torch.tensor(image_resized.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict and resize back to original dimensions
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = cv2.resize(pred_mask, (original_shape[1], original_shape[0]))
        pred_mask = (pred_mask > 0.5).astype(np.float32)

    # Decode all Ground Truth RLEs for this image and combine
    img_solutions = solution_df[solution_df['ImageId'] == img_id]['EncodedPixels'].tolist()
    gt_mask = np.zeros(original_shape, dtype=np.float32)
    for rle in img_solutions:
        single_mask = rle_decode(rle, (original_shape[1], original_shape[0]))
        gt_mask = np.maximum(gt_mask, single_mask)

    # Calculate metrics
    iou, dice = calculate_metrics(pred_mask, gt_mask)
    ious.append(iou)
    dices.append(dice)

# 5. Save Results as Image
mean_iou = np.mean(ious)
mean_dice = np.mean(dices)

print(f"Mean IoU: {mean_iou:.4f}")
print(f"Mean Dice Coefficient: {mean_dice:.4f}")

plt.figure(figsize=(8, 6))
bars = plt.bar(['Mean IoU', 'Mean Dice Coefficient'], [mean_iou, mean_dice], color=['#1f77b4', '#ff7f0e'])
plt.ylim(0, 1.0)
plt.title('U-Net Performance on Stage 1 Test Set')
plt.ylabel('Score (0.0 to 1.0)')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom', fontweight='bold')

plt.savefig('test_metrics_summary.png')
print("Saved metric visualization to test_metrics_summary.png")
