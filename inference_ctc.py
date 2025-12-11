import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import math
from collections import deque
from itertools import groupby

# Re-define Model (Must match training exactly)
# Or import it if you structure your code to allow imports (recommended)
# For now, I'll assume we can import from train_ctc if it's in the same folder.
try:
    from train_ctc import SignCTCModel
    from dataset_synthetic import VideoLoader # To re-use classes if needed? Or just hardcode logic.
except ImportError:
    # If standard import fails (e.g. running as script)
    pass

# Configuration
MODEL_PATH = "transformer_ctc_refined.pth"
DATA_DIR = "data"
FEATURES_PER_FRAME = 126
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mapping: Need to know classes to index
def get_classes():
    # Hardcoded to match the trained model 'transformer_ctc_refined.pth'
    return ['a', 'hello', 'm', 'name', 'nothing', 'person', 'what']

def greedy_decode_ctc(logits, vocab, blank_idx):
    """
    logits: [SeqLen, B=1, NumClasses]
    Returns string of decoded labels.
    """
    # 1. Argmax
    probs = torch.softmax(logits, dim=2)
    max_probs, indices = torch.max(probs, dim=2)
    indices = indices.squeeze(1).cpu().numpy() # [SeqLen]
    
    # 2. Collapse Repeats & Remove Blanks
    decoded_inds = []
    for i, (k, g) in enumerate(groupby(indices)):
        if k != blank_idx:
            decoded_inds.append(k)
            
    # 3. Map to classes
    result = [vocab[i] for i in decoded_inds if vocab[i] != 'nothing']
    return " ".join(result)

def main():
    classes = get_classes()
    vocab_size = len(classes)
    blank_idx = vocab_size # Last index
    print(f"Vocab: {classes}, Blank: {blank_idx}")
    
    # Load Model
    # Note: Architecture must match train_ctc.py
    # If you moved SignCTCModel to a shared file, import it.
    # Otherwise, copy-paste class definition here to be safe.
    # (Assuming import from train_ctc works for now as it's in same dir)
    from train_ctc import SignCTCModel 

    model = SignCTCModel(
        vocab_size=vocab_size,
        input_dim=FEATURES_PER_FRAME,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # MediaPipe
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    
    # Buffer: We need to maintain a growing buffer or a sliding window of features
    # For "Dictation", infinite growing buffer is bad.
    # Let's do a sliding window of recent history (e.g., 5 seconds = 150 frames)
    # And decode that window repeatedly.
    WINDOW_LEN = 150
    feature_buffer = deque(maxlen=WINDOW_LEN)
    
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)
        
        # Extract features
        frame_kps = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    frame_kps.extend([lm.x, lm.y, lm.z])
            if len(results.multi_hand_landmarks) < 2:
                frame_kps.extend([0.0] * (2 - len(results.multi_hand_landmarks)) * 63)
        else:
            frame_kps.extend([0.0] * 126)
            
        feature_buffer.append(frame_kps[:126])
        
        if len(feature_buffer) > 2:
            input_tensor = torch.tensor(list(feature_buffer), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(input_tensor)
            raw_text = greedy_decode_ctc(logits, classes, blank_idx)
            
            # Stabilization/Debouncing
            # We add the raw text to a history buffer
            if not hasattr(main, "text_history"):
                main.text_history = deque(maxlen=10)
            main.text_history.append(raw_text)
            
            # Find the most common text in history
            from collections import Counter
            c = Counter(main.text_history)
            stable_text, count = c.most_common(1)[0]
            
            # Only update if consistent
            if count > 6: # 6/10 frames agree
                decoded_text = stable_text
            else:
                decoded_text = "..." # Unstable
        else:
            decoded_text = ""
        
        # UI
        cv2.putText(frame, f"Decoded: {decoded_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("CTC Real-time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
