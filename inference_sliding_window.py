import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import math
import argparse
from collections import deque, Counter

# Configuration
MODEL_PATH = "transformer_sliding_window.pth"
DATA_DIR = "data"
WINDOW_SIZE = 30 # Must match model training length
STRIDE = 5       # Run inference every N frames
HISTORY_LENGTH = 5 # K = size of smoothing history
CONFIDENCE_THRESHOLD = 0.7
FEATURES_PER_FRAME = 126

# Model Hyperparameters (Must match training)
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Definition (Copy from inference_transformer.py) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SignLanguageTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=126, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.5):
        super(SignLanguageTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=WINDOW_SIZE)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1) 
        output = self.dropout(output)
        output = self.fc_out(output)
        return output

# --- Inference Utils ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    frame_keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                frame_keypoints.extend([lm.x, lm.y, lm.z])
        
        detected_hands = len(results.multi_hand_landmarks)
        if detected_hands < 2:
            padding_needed = (2 - detected_hands) * 21 * 3
            frame_keypoints.extend([0.0] * padding_needed)
    else:
        # If no hands, fill with zeros
        frame_keypoints.extend([0.0] * FEATURES_PER_FRAME)
    return frame_keypoints[:FEATURES_PER_FRAME]

def get_classes():
    if os.path.exists(DATA_DIR):
        classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
        return classes
    return []

def main():
    classes = get_classes()
    print(f"Classes: {classes}")
    
    # Load Model
    model = SignLanguageTransformer(
        num_classes=len(classes),
        input_dim=FEATURES_PER_FRAME,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model.eval()

    # buffers
    # 1. Input Buffer: Stores landmark vectors
    landmark_buffer = deque(maxlen=WINDOW_SIZE)
    
    # 2. Prediction History: Stores (class_index, confidence)
    prediction_history = deque(maxlen=HISTORY_LENGTH)
    
    # State tracking
    current_stable_label = "Waiting..."
    current_stable_conf = 0.0
    
    cap = cv2.VideoCapture(0)
    
    # Match dataset creator resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Camera started. Smoothing window: {HISTORY_LENGTH}, Stride: {STRIDE}")
    
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam. Exiting loop.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # 1. Visualization
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 2. Extract Keypoints
        keypoints = extract_keypoints(results)
        
        # 3. Update Buffer
        landmark_buffer.append(keypoints)
        
        # 4. Check if we should run inference
        # Run only if buffer is full AND stride condition is met
        if len(landmark_buffer) == WINDOW_SIZE and (frame_idx % STRIDE == 0):
            # Prepare Input
            input_tensor = torch.tensor([list(landmark_buffer)], dtype=torch.float32).to(device)
            # Shape: [1, WINDOW_SIZE, FEATURES_PER_FRAME]
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                # Get top prediction
                conf, pred_idx = torch.max(probs, dim=1)
                conf = conf.item()
                pred_idx = pred_idx.item()
                
                # Add to history
                prediction_history.append((pred_idx, conf))
                
                # 5. Smoothing / Stabilization Logic
                if len(prediction_history) == HISTORY_LENGTH:
                    # Count occurrences of class indices in history
                    # We only count a prediction if its confidence > CONFIDENCE_THRESHOLD?
                    # Or count everything and check avg confidence?
                    # User said: "Same class appears in the majority... AND its softmax prob is above threshold"
                    
                    # Let's filter history by threshold first
                    valid_preds = [p for p in prediction_history if p[1] > CONFIDENCE_THRESHOLD]
                    
                    if valid_preds:
                        # Most common class
                        indices = [p[0] for p in valid_preds]
                        most_common = Counter(indices).most_common(1)
                        
                        if most_common:
                            top_class_idx, count = most_common[0]
                            
                            # Check majority (e.g., > K/2 or just simply most common?)
                            # User said "majority of the last K". Let's assume > K/2 strictly
                            if count > (HISTORY_LENGTH / 2):
                                new_label = classes[top_class_idx]
                                
                                # Debouncing: Update only if changed
                                if new_label != current_stable_label:
                                    current_stable_label = new_label
                                    current_stable_conf = 0.0 # Just for display, or avg valid conf?
                                    # Let's average the confidence of the winning class
                                    winning_confs = [p[1] for p in valid_preds if p[0] == top_class_idx]
                                    current_stable_conf = sum(winning_confs) / len(winning_confs)
                                    print(f"New Stable Label: {current_stable_label} ({current_stable_conf:.2f})")
                                else:
                                    # Just update confidence
                                    winning_confs = [p[1] for p in valid_preds if p[0] == top_class_idx]
                                    current_stable_conf = sum(winning_confs) / len(winning_confs)

        # UI Display
        cv2.putText(frame, f"Prediction: {current_stable_label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {current_stable_conf:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Sliding Window Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("debug_error.txt", "w") as f:
            f.write(traceback.format_exc())
        print("Error logged to debug_error.txt")
