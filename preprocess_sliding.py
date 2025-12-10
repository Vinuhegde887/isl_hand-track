import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Configuration
DATA_DIR = "data"
OUTPUT_CSV = "keypoints_sliding_data.csv"
SEQUENCE_LENGTH = 30
STRIDE = 10 # Extract a window every 10 frames (Results in 3x more data approx)
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
    frame_keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                frame_keypoints.extend([lm.x, lm.y, lm.z])
        detected_hands = len(results.multi_hand_landmarks)
        if detected_hands < NUM_HANDS:
            padding_needed = (NUM_HANDS - detected_hands) * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
            frame_keypoints.extend([0.0] * padding_needed)
    else:
        frame_keypoints.extend([0.0] * FEATURES_PER_FRAME)
    return frame_keypoints[:FEATURES_PER_FRAME]

def process_video(video_path):
    """
    Reads entire video, extracts all frame keypoints.
    Returns list of keypoint vectors.
    """
    cap = cv2.VideoCapture(video_path)
    video_keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        keypoints = extract_keypoints(results)
        video_keypoints.append(keypoints)
            
    cap.release()
    return video_keypoints

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        return

    all_sequences = []
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
                # 1. Get all frames
                full_video_data = process_video(video_path)
                total_frames = len(full_video_data)
                
                # 2. Extract Sliding Windows
                if total_frames < SEQUENCE_LENGTH:
                    # Pad short video once
                    padding = [ [0.0]*FEATURES_PER_FRAME ] * (SEQUENCE_LENGTH - total_frames)
                    seq = full_video_data + padding
                    all_sequences.append([item for sublist in seq for item in sublist])
                    labels.append(class_name)
                else:
                    # Slide window
                    for start_idx in range(0, total_frames - SEQUENCE_LENGTH + 1, STRIDE):
                        end_idx = start_idx + SEQUENCE_LENGTH
                        window = full_video_data[start_idx:end_idx]
                        
                        # Flatten
                        flattened_window = [item for sublist in window for item in sublist]
                        all_sequences.append(flattened_window)
                        labels.append(class_name)
                        
            except Exception as e:
                print(f"Error processing {video_file}: {e}")

    print(f"Total sequences generated: {len(all_sequences)}")
    print("Creating DataFrame...")
    df = pd.DataFrame(all_sequences)
    df['label'] = labels
    
    print(f"Saving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
