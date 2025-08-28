# realtime_vein_overlay.py
import cv2
import numpy as np
import time

# ====== 1) Choose your video source ======
# A) Webcam: use 0 or 1
SOURCE = 0

# B) DroidCam (replace with your phone IP):
# SOURCE = "http://192.168.0.123:4747/video"   # typical DroidCam URL

cap = cv2.VideoCapture(SOURCE)

# If using IP stream, improve buffering
if isinstance(SOURCE, str):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise SystemExit("❌ Could not open video source. Check SOURCE value.")

print("✅ Running. Keys: q=quit, s=save, +/- adjust sensitivity, [ ] adjust thickness.")

# Tunables (can be changed live with keys below)
canny_lo, canny_hi = 30, 100  # edge sensitivity
dilate_size = 2               # vein thickness (2–4 good)
tophat_size = 15              # background suppression kernel (odd number)

frame_idx = 0
last_save = time.time()

def odd(n): return n if n % 2 else n+1

while True:
    ok, frame = cap.read()
    if not ok:
        # brief retry for IP streams
        time.sleep(0.01)
        continue

    # ---------- Preprocess ----------
    # Resize for speed (change to 1.0 for native size)
    scale = 0.8
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Optional denoise that preserves edges
    # bilateral is fast and keeps vein edges better than Gaussian
    den = cv2.bilateralFilter(frame, d=5, sigmaColor=60, sigmaSpace=60)

    gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)

    # Local contrast (CLAHE) to make veins pop
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    con = clahe.apply(gray)

    # Illumination / skin suppression: top-hat (remove smooth background)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (odd(tophat_size), odd(tophat_size)))
    tophat = cv2.morphologyEx(con, cv2.MORPH_TOPHAT, kernel)

    # Edge emphasis -> candidate veins
    edges = cv2.Canny(tophat, canny_lo, canny_hi)

    # Remove small speckles (opening) and strengthen lines (dilate)
    opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    dil = cv2.dilate(opened, np.ones((dilate_size, dilate_size), np.uint8), iterations=1)

    # Overlay red veins on original
    overlay = frame.copy()
    overlay[dil > 0] = (0, 0, 255)

    # Small HUD
    hud = overlay.copy()
    text = f"lo/hi:{canny_lo}/{canny_hi} | thick:{dilate_size} | tophat:{tophat_size}"
    cv2.putText(hud, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Vein Overlay (red) — q quit, s save, +/- sensitivity, [ ] thickness", hud)
    cv2.imshow("Intermediate: TopHat", tophat)
    cv2.imshow("Intermediate: Edges", edges)

    # ---------- Key controls ----------
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        now = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"vein_overlay_{now}.png", hud)
        cv2.imwrite(f"vein_mask_{now}.png", dil)
        print(f"💾 saved vein_overlay_{now}.png & vein_mask_{now}.png")
        last_save = time.time()
    elif k in (ord('+'), ord('=')):        # increase sensitivity
        canny_lo = max(0, canny_lo - 2)
        canny_hi = max(canny_lo + 10, canny_hi - 4)
    elif k == ord('-'):                    # decrease sensitivity
        canny_lo = min(100, canny_lo + 2)
        canny_hi = min(250, canny_hi + 4)
    elif k == ord('['):                    # thinner lines
        dilate_size = max(1, dilate_size - 1)
    elif k == ord(']'):                    # thicker lines
        dilate_size = min(7, dilate_size + 1)
    elif k == ord(','):                    # smaller tophat window
        tophat_size = max(7, tophat_size - 2)
    elif k == ord('.'):                    # larger tophat window
        tophat_size = min(31, tophat_size + 2)

cap.release()
cv2.destroyAllWindows()
# End of realtime_vein_overlay.py