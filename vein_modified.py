import cv2
import numpy as np

# -----------------------------
# 🔹 Step 1: Read Frame from Camera
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
# 🔹 Step 5: Real-time Vein Pipeline
# -----------------------------
def real_time_vein_detection():
    cap = cv2.VideoCapture("http://192.168.1.6:4747/video")

 # default camera (laptop/phone webcam)
    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    print("Press 'q' to exit the real-time vein detection.")

    frame_count = 0  # Initialize frame counter

    while True:
        frame = get_camera_frame(cap)
        if frame is None:
            break

        enhanced = enhance_veins(frame)
        multi_scale = multi_scale_detect(enhanced)
        final_result = wide_kernel_filter(multi_scale)

        # Show original and processed side by side
        combined_display = np.hstack((frame, final_result))
        cv2.imshow("Vein Detection (Original | Processed)", combined_display)

        # ---------------- Screenshot on 's' key ----------------
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(f"screenshots/vein_{frame_count}.png", final_result)
            frame_count += 1
            print(f"Saved screenshot {frame_count}!")
            
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# 🔹 Example Usage
# -----------------------------
if __name__ == "__main__":
    real_time_vein_detection()
