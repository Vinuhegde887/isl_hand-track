import cv2
import mediapipe as mp
import os
import sys

# Configuration
INPUT_DIR = "data"
OUTPUT_DIR = "data_annotated"
NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Initialize MediaPipe Hands and Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Fallback for FPS if not detected correctly
    if fps == 0 or fps is None:
        fps = 30.0

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use mp4v for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()
    # print(f"Processed: {os.path.basename(input_path)} -> {frame_count} frames")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    print(f"Found classes: {classes}")

    for class_name in classes:
        class_input_dir = os.path.join(INPUT_DIR, class_name)
        class_output_dir = os.path.join(OUTPUT_DIR, class_name)
        
        # Create class-specific output directory
        os.makedirs(class_output_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(class_input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"Annotating class '{class_name}': {len(video_files)} videos...")
        
        for video_file in video_files:
            input_path = os.path.join(class_input_dir, video_file)
            output_path = os.path.join(class_output_dir, video_file)
            
            # Skip if already exists? (Optional, currently overwriting)
            # if os.path.exists(output_path): continue
            
            try:
                process_video(input_path, output_path)
            except Exception as e:
                print(f"Failed to process {video_file}: {e}")

    print("All videos processed and saved to 'data_annotated/'.")

if __name__ == "__main__":
    main()
