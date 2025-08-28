import cv2
import numpy as np
import os

# -----------------------------
# 🔹 Step 0: Create screenshots folder if not exists
# -----------------------------
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# -----------------------------
# 🔹 Step 1: Read frame from camera
# -----------------------------
def get_camera_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

# -----------------------------
# 🔹 Step 2: DiffVein-inspired Enhancement
# -----------------------------
def enhance_veins(frame):
    blur = cv2.GaussianBlur(frame, (5,5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blur)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

# -----------------------------
# 🔹 Step 3: GLVM-inspired Multi-scale Detection
# -----------------------------
def multi_scale_detect(frame):
    small = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    edges_global = cv2.Canny(small, 50, 150)
    edges_global = cv2.resize(edges_global, (frame.shape[1], frame.shape[0]))
    edges_local = cv2.Canny(frame, 100, 200)
    combined = cv2.addWeighted(edges_global, 0.5, edges_local, 0.5, 0)
    return combined

# -----------------------------
# 🔹 Step 4: StarLKNet-inspired Wide Kernel Filter
# -----------------------------
def wide_kernel_filter(frame):
    filtered = cv2.medianBlur(frame, 7)
    return filtered

# -----------------------------
# 🔹 Step 5: Real-time vein detection
# -----------------------------
def real_time_vein_detection():
    # Try USB/webcam first
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("USB/Webcam not found, trying DroidCam Wi-Fi...")
        cap = cv2.VideoCapture("http://192.168.1.6:4747/video")
        if not cap.isOpened():
            print("Error: Cannot access camera")
            return
        else:
            print("Camera opened successfully!")
    else:
        print("Camera opened successfully!")

    print("Press 's' to save screenshot, 'q' to exit.")

    frame_count = 0
    while True:
        frame = get_camera_frame(cap)
        if frame is None:
            break

        enhanced = enhance_veins(frame)
        multi_scale = multi_scale_detect(enhanced)
        final_result = wide_kernel_filter(multi_scale)

        # Side-by-side display
        combined_display = np.hstack((frame, final_result))
        cv2.imshow("Vein Detection (Original | Processed)", combined_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_path = os.path.join(os.getcwd(), "screenshots", f"vein_{frame_count}.png")
            cv2.imwrite(save_path, final_result)
            frame_count += 1
            print(f"Saved screenshot at: {save_path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# 🔹 Step 6: Run
# -----------------------------
if __name__ == "__main__":
    real_time_vein_detection()
