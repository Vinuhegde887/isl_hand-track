import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Configuration
DATA_DIR = "data"
OUTPUT_CSV = "keypoints_data.csv"
SEQUENCE_LENGTH = 30
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_FRAME = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK # 126

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=NUM_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    """
    Extracts keypoints from MediaPipe results.
    Returns a flattened list of shape (FEATURES_PER_FRAME,).
    """
    frame_keypoints = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                frame_keypoints.extend([lm.x, lm.y, lm.z])
        
        # If fewer hands than expected were detected, pad with zeros
        detected_hands = len(results.multi_hand_landmarks)
        if detected_hands < NUM_HANDS:
            padding_needed = (NUM_HANDS - detected_hands) * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
            frame_keypoints.extend([0.0] * padding_needed)
            
        # If more hands (unlikely due to max_num_hands setting, but for safety), truncate
        # This part is actually handled by max_num_hands in mp.Hands, but good to be aware.
        
    else:
        # No hands detected, pad with all zeros
        frame_keypoints.extend([0.0] * FEATURES_PER_FRAME)
        
    return frame_keypoints[:FEATURES_PER_FRAME]

def process_video(video_path, sequence_length):
    """
    Processes a single video file and returns a list of frames (each frame is a list of features).
    Returns shape (SEQUENCE_LENGTH, FEATURES_PER_FRAME).
    """
    cap = cv2.VideoCapture(video_path)
    video_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        video_data.append(keypoints)
        
        if len(video_data) == sequence_length:
            break
            
    cap.release()
    
    # Pad if video is shorter than sequence length
    if len(video_data) < sequence_length:
        padding_frames = sequence_length - len(video_data)
        zeros_frame = [0.0] * FEATURES_PER_FRAME
        for _ in range(padding_frames):
            video_data.append(zeros_frame)
            
    return video_data

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        return

    all_data = []
    labels = []
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"Found classes: {classes}")

    for class_name in classes:
        class_path = os.path.join(DATA_DIR, class_name)
        video_files = [f for f in os.listdir(class_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"Processing class '{class_name}': {len(video_files)} videos")
        
        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)
            try:
                # Process video -> (50, 126)
                sequence_data = process_video(video_path, SEQUENCE_LENGTH)
                
                # Flatten sequence for CSV row: (50 * 126,)
                flattened_sequence = [item for sublist in sequence_data for item in sublist]
                
                all_data.append(flattened_sequence)
                labels.append(class_name)
                
            except Exception as e:
                print(f"Error processing {video_file}: {e}")

    print("Creating DataFrame...")
    # Create DataFrame
    # Columns will be 0, 1, 2, ... 6299
    df = pd.DataFrame(all_data)
    df['label'] = labels
    
    output_path =  OUTPUT_CSV
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
