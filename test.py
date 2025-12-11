import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import time
import math
from inference_emitter import InferenceEmitter

# Configuration
MODEL_PATH = "transformer_sliding_window.pth"
DATA_DIR = "data"
WINDOW_SIZE = 30 
STRIDE = 1
FEATURES_PER_FRAME = 126
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Re-define Model Architecture
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

def extract_keypoints(results):
    frame_keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                frame_keypoints.extend([lm.x, lm.y, lm.z])
        # Padding
        if len(results.multi_hand_landmarks) < 2:
            frame_keypoints.extend([0.0] * (2 - len(results.multi_hand_landmarks)) * 63)
    else:
        frame_keypoints.extend([0.0] * FEATURES_PER_FRAME)
    return frame_keypoints[:FEATURES_PER_FRAME]

def get_classes():
    if os.path.exists(DATA_DIR):
        classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
        return classes
    return []

def main():
    print("Initializing Simple Inference Test...")
    classes = get_classes()
    if not classes:
        print("No classes found in data directory.")
        return
        
    print(f"Classes: {classes}")
    
    # Load Model
    model = SignLanguageTransformer(len(classes)).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Model file {MODEL_PATH} not found.")
        return
        
    # Init Emitter
    emitter = InferenceEmitter(
        model=model,
        label_map=classes,
        fps=30,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        device=DEVICE,
        debounce_k=3,
        conf_min=0.6
    )

    # MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    print("Camera started. Press 'q' to quit.")
    
    start_time_real = time.time()
    last_final_token = "..."
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Draw
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Extract
        kps = extract_keypoints(results)
        timestamp = time.time() - start_time_real
        
        # Emit logic
        finalized_event = emitter.process_frame(kps, timestamp)
        
        if finalized_event:
            token = finalized_event['token']
            conf = finalized_event['avg_conf']
            last_final_token = f"{token} ({conf:.2f})"
            print(f"Detected: {last_final_token}")
            
        current_active = emitter.current_token if emitter.current_token else "..."

        # UI
        cv2.putText(frame, f"Live: {current_active}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('Simple Test', frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
