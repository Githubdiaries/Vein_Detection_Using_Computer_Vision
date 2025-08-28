import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize 

# ---------- 1. Load Image ----------
img = cv2.imread("hand.jpg")       # replace with your captured hand image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- 2. Denoise ----------
denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

# ---------- 3. Enhance Contrast (CLAHE) ----------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
contrast = clahe.apply(denoised)

# ---------- 4. Suppress Skin / Highlight Veins ----------
# Method A: Frangi filter (tubular structures like veins)
vein_frangi = frangi(contrast.astype(np.float32)/255.0)
vein_frangi = (vein_frangi * 255).astype(np.uint8)

# Method B: Morphological Top-hat (removes smooth background)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
vein_tophat = cv2.morphologyEx(contrast, cv2.MORPH_TOPHAT, kernel)

# Combine both methods
vein_enhanced = cv2.addWeighted(vein_frangi, 0.6, vein_tophat, 0.4, 0)

# ---------- 5. Adaptive Threshold ----------
binary = cv2.adaptiveThreshold(vein_enhanced, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 2)

# ---------- 6. Skeletonize (optional) ----------
skeleton = skeletonize(binary > 0)
skeleton = (skeleton * 255).astype(np.uint8)

# ---------- 7. Overlay veins on original hand ----------
overlay = img.copy()
overlay[skeleton > 0] = (0,0,255)   # red veins

# ---------- 8. Save Results ----------
cv2.imwrite("01_gray.png", gray)
cv2.imwrite("02_contrast.png", contrast)
cv2.imwrite("03_vein_frangi.png", vein_frangi)
cv2.imwrite("04_vein_tophat.png", vein_tophat)
cv2.imwrite("05_vein_enhanced.png", vein_enhanced)
cv2.imwrite("06_binary.png", binary)
cv2.imwrite("07_skeleton.png", skeleton)
cv2.imwrite("08_overlay.png", overlay)

print("✅ Vein detection pipeline complete. Check saved images.")
# ---------- 9. Display Results ---------- 