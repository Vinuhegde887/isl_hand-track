import os
import time
import cv2
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from camera import CameraController

def get_next_filename(folder_path):
    existing = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    if not existing:
        return "0001.mp4"
    existing_numbers = [int(f.split(".")[0]) for f in existing if f.split(".")[0].isdigit()]
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    return f"{next_number:04d}.mp4"

# ---------- 1.  PRE-PROCESSING HELPERS ----------
def build_clahe_grid(gray, clip=3.0, grid=8):
    """local contrast enhancement"""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)

def gamma_correct(img, gamma=0.8):
    """raise mid-tones (hand surface) without blowing highlights"""
    inv = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img, table)

def mild_gaussian_blur(img, k=3):
    """kill high-freq sensor noise while keeping edges"""
    return cv2.GaussianBlur(img, (k, k), 0)

def record_clip(cam, folder_path, duration=2):
    """Record a 2-second clip using CameraController's built-in methods."""
    filename = os.path.join(folder_path, get_next_filename(folder_path))
    
    # Start recording using the class method
    cam.start_recording(filename)
    
    # Record for specified duration
    time.sleep(duration)
    
    # Stop recording using the class method
    cam.stop_recording()
    print(f"? Saved: {filename}")
    
    # Restart camera to reset state for preview
    cam.picam2.stop()
    time.sleep(0.2)
    cam.picam2.start()
    time.sleep(0.3)

def main():
    class_label = input("Enter class label: ").strip()
    folder_path = os.path.join("dataset", class_label)
    os.makedirs(folder_path, exist_ok=True)
    
    cam = CameraController(resolution=(1280, 960), framerate=60)
    cam.start()
    
    print("Camera started. Press 'q' to record 2-second clip, ESC to exit.")
    
    while True:
        # Get and display frame
        frame = cam.get_frame()
         # ---------- 3.  IMPROVED IR PIPE ----------
        red = frame[:, :, 2]                                    # least IR pollution
        red = build_clahe_grid(red, clip=3.0, grid=8)          # local contrast
        red = gamma_correct(red, gamma=0.8)                    # raise hand tones
        red = mild_gaussian_blur(red, k=3)                     # denoise
        pseudo = cv2.merge([red, red, red])                    # 3-ch RGB MediaPipe wants
        # --------------------------------------
        
        cv2.imshow("Camera Preview", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            if not cam.is_recording:  # Only start if not already recording
                print("Recording...")
                record_clip(cam, folder_path, duration=2)
                print("Ready for next recording!")
        
        if key == 27:  # ESC
            break
    
    cam.stop()
    cv2.destroyAllWindows()
    print("Camera stopped. Exiting.")

if __name__ == "__main__":
    main()
