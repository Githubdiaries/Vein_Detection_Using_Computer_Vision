import cv2
import numpy as np
import os
os.chdir(r"C:\Users\aksas\OneDrive\Desktop\Vein_Detection_Using_Computer_Vision\screenshots")


# ---------- 1. Load image ----------
img = cv2.imread("vein_0.png")


# replace with your image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- 2. Preprocessing ----------
# Denoise
denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

# Contrast Enhancement (CLAHE)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
contrast = clahe.apply(denoised)

# ---------- 3. Highlight veins ----------
# Top-hat filtering (extract dark thin structures)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
tophat = cv2.morphologyEx(contrast, cv2.MORPH_TOPHAT, kernel)

# Edge emphasis
edges = cv2.Canny(tophat, 30, 100)

# ---------- 4. Noise cleanup ----------
# Morphological opening removes speckles
cleaned = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

# Strengthen veins (dilate lines slightly)
veins = cv2.dilate(cleaned, np.ones((2,2), np.uint8), iterations=1)

# ---------- 5. Overlay on original ----------
overlay = img.copy()
overlay[veins > 0] = (0,0,255)   # draw detected veins in RED

# ---------- 6. Save results ----------
cv2.imwrite("01_gray.png", gray)
cv2.imwrite("02_contrast.png", contrast)
cv2.imwrite("03_tophat.png", tophat)
cv2.imwrite("04_edges.png", edges)
cv2.imwrite("05_cleaned.png", cleaned)
cv2.imwrite("06_veins.png", veins)
cv2.imwrite("07_overlay.png", overlay)

print("✅ Vein overlay created: check 07_overlay.png")
# ---------- 7. Display results ----------