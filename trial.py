import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Load image
img = cv2.imread("hand.jpg")   # replace with your image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 1: Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Extract green channel (veins are darker here)
green = img[:, :, 1]

# Step 3: CLAHE (contrast enhancement)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
green_clahe = clahe.apply(green)

# Step 4: Gaussian blur (to reduce noise)
blur = cv2.GaussianBlur(green_clahe, (5, 5), 0)

# Step 5: Edge detection (Canny)
edges = cv2.Canny(blur, 30, 80)

# Show results
titles = ['Original', 'Gray', 'Green Channel', 'CLAHE + Blur', 'Edges']
images = [img_rgb, gray, green, blur, edges]

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
