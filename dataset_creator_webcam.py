import os
import time
import cv2
import numpy as np

def get_next_filename(folder_path):
    existing = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    if not existing:
        return "0001.mp4"
    existing_numbers = [int(f.split(".")[0]) for f in existing if f.split(".")[0].isdigit()]
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    return f"{next_number:04d}.mp4"

def record_clip(cap, folder_path, duration=2.0, fps=30.0, frame_size=(640, 480)):
    """Record a clip for a specific duration using OpenCV VideoWriter."""
    filename = os.path.join(folder_path, get_next_filename(folder_path))
    
    # Define codec and create VideoWriter object
    # mp4v is a common option, usually works on Windows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(filename, fourcc, fps, frame_size)
    
    print(f"Recording to {filename}...")
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Camera Preview', frame)
            cv2.waitKey(1) # Keep UI responsive
        else:
            break
            
    out.release()
    print(f"? Saved: {filename}")

def main():
    class_label = input("Enter class label: ").strip()
    folder_path = os.path.join("data", class_label)
    os.makedirs(folder_path, exist_ok=True)
    
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Try to set resolution (optional, might default to 640x480)
    width = 1280
    height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Read actual width/height/fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0
        
    print(f"Camera started: {width}x{height} @ {fps}fps")
    print("Press 'q' to record 2-second clip, ESC to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            cv2.imshow("Camera Preview", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Pause preview logic slightly to handle recording in the same loop or separate
                # Logic here is simple: blocking record
                record_clip(cap, folder_path, duration=2, fps=fps, frame_size=(width, height))
                print("Ready for next recording!")
            
            elif key == 27:  # ESC
                break
                
    except KeyboardInterrupt:
        pass
        
    cap.release()
    cv2.destroyAllWindows()
    print("Camera stopped. Exiting.")

if __name__ == "__main__":
    main()
