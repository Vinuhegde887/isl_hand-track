import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import math
import argparse

# Configuration
MODEL_PATH = "transformer_action.pth"
DATA_DIR = "data" 
SEQUENCE_LENGTH = 30 # Match training
FEATURES_PER_FRAME = 126

# Model Hyperparameters (Must match training)
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Definition (Copy from train_transformer.py) ---
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
        self.pos_encoder = PositionalEncoding(d_model, max_len=SEQUENCE_LENGTH)
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
    static_image_mode=False,
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
        frame_keypoints.extend([0.0] * FEATURES_PER_FRAME)
    return frame_keypoints[:FEATURES_PER_FRAME]

def get_classes():
    if os.path.exists(DATA_DIR):
        classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
        return classes
    return ['hello', 'thank you'] # Fallback

def main():
    parser = argparse.ArgumentParser(description="Run Transformer Inference on a video.")
    args = parser.parse_args()

    classes = get_classes()
    print(f"Classes: {classes}")
    
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

    cap = cv2.VideoCapture(0) # Use webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # User Interface States
    STATE_WAITING = 0
    STATE_RECORDING = 2 # Kept ID for consistency
    STATE_RESULT = 3
    
    current_state = STATE_WAITING
    
    # State variables
    sequence = []
    hand_detected_frames = 0
    result_start_time = 0
    last_prediction = ""
    last_confidence = 0.0
    
    HAND_DETECTION_THRESHOLD = 5 # Reduced threshold for faster response
    RESULT_DURATION = 2.0 # Seconds

    print("Camera started. Show hands to automatically start prediction.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Check for hands
        hands_present = False
        if results.multi_hand_landmarks:
            hands_present = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # State Machine
        if current_state == STATE_WAITING:
            cv2.putText(frame, "WAITING FOR HANDS...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if hands_present:
                hand_detected_frames += 1
                if hand_detected_frames >= HAND_DETECTION_THRESHOLD:
                    current_state = STATE_RECORDING
                    sequence = []
                    hand_detected_frames = 0
            else:
                hand_detected_frames = 0
                            
        elif current_state == STATE_RECORDING:
            cv2.putText(frame, "RECORDING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"{len(sequence)}/{SEQUENCE_LENGTH}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            
            if len(sequence) == SEQUENCE_LENGTH:
                # Predict
                print("Predicting...")
                with torch.no_grad():
                    input_tensor = torch.tensor([sequence], dtype=torch.float32).to(device)
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    best_idx = torch.argmax(probs, dim=1).item()
                    last_confidence = probs[0][best_idx].item()
                    last_prediction = classes[best_idx]
                
                print(f"Result: {last_prediction} ({last_confidence:.2f})")
                current_state = STATE_RESULT
                result_start_time = cv2.getTickCount()
                
        elif current_state == STATE_RESULT:
            elapsed_time = (cv2.getTickCount() - result_start_time) / cv2.getTickFrequency()
            
            # UI
            color = (0, 255, 0) if last_confidence > 0.6 else (0, 165, 255)
            cv2.putText(frame, f"PREDICTION: {last_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"CONFIDENCE: {last_confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if elapsed_time > RESULT_DURATION:
                current_state = STATE_WAITING
                hand_detected_frames = 0

        cv2.imshow('Transformer Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
