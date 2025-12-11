import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import time
import math
import threading
from inference_emitter import InferenceEmitter
from gemini_client import call_gemini

# Configuration
MODEL_PATH = "transformer_sliding_window.pth"
DATA_DIR = "data"
WINDOW_SIZE = 30 
STRIDE = 1       # Emitter handles stride internally if needed
FEATURES_PER_FRAME = 126
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Thresholds
SILENCE_THRESHOLD = 1.0  # Seconds of silence to trigger translation

# Re-define Model Architecture (Must match training exactly)
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

# Thread worker for Gemini
def gemini_worker(tokens, confs, callback):
    try:
        # print(f"Sending to Gemini: {tokens}")
        result = call_gemini(tokens, confs)
        callback(result)
    except Exception as e:
        callback(f"Error: {e}")

def main():
    classes = get_classes()
    if not classes:
        print("No classes found in data directory.")
        return
        
    print(f"Classes: {classes}")
    
    # Load Model
    model = SignLanguageTransformer(len(classes)).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Init Emitter
    emitter = InferenceEmitter(
        model=model,
        label_map=classes,
        fps=30,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        device=DEVICE,
        debounce_k=3,    # Needs 3 consecutive consistent frames
        conf_min=0.6     # Min confidence
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Camera started.")
    
    start_time_real = time.time()
    
    # State for Sentence Construction
    draft_tokens = []     # ["Hello", "What"]
    draft_confs = []      # [0.99, 0.85]
    final_sentence = ""   # "Hello, what is your name?"
    
    last_action_time = time.time()
    is_translating = False
    
    # Callback to update UI from thread
    def on_translation_done(text):
        nonlocal final_sentence, is_translating
        final_sentence = text
        is_translating = False
        print(f"Gemini: {text}")

    try:
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
            
            # Emit Logic
            finalized_event = emitter.process_frame(kps, timestamp)
            
            # 1. Handle New Token
            if finalized_event:
                token = finalized_event['token']
                conf = finalized_event['avg_conf']
                
                # Exclude specific noise tokens if needed
                if token not in ["nothing", "background"]:
                    # If we have a previously finalized sentence displayed, clears it when new sign starts
                    # Logic: If draft is empty, we are starting a NEW thought, so clear the OLD final sentence
                    if not draft_tokens: 
                        final_sentence = "" 
                        
                    draft_tokens.append(token)
                    draft_confs.append(conf)
                    last_action_time = time.time()
                    print(f"Added token: {token}")

            # 2. Check Silence / Trigger Translation
            if len(draft_tokens) > 0 and not is_translating:
                time_since_last = time.time() - last_action_time
                
                if time_since_last > SILENCE_THRESHOLD:
                    # Trigger Translation
                    print("Silence detected. Triggering Gemini...")
                    is_translating = True
                    
                    # Create thread copy of data
                    tokens_to_send = list(draft_tokens)
                    confs_to_send = list(draft_confs)
                    
                    # CLEAR BUFFER IMMEDIATELY (Handover)
                    draft_tokens.clear()
                    draft_confs.clear()
                    
                    t = threading.Thread(target=gemini_worker, args=(tokens_to_send, confs_to_send, on_translation_done))
                    t.start()
            
            # 3. UI Rendering
            # Active recognized token (live, unstable)
            active_tk = emitter.current_token if emitter.current_token else ""
            
            # Top Bar: Status
            status_text = "Listening..."
            if is_translating:
                status_text = "Translating..."
            elif final_sentence:
                status_text = "Complete"
                
            # Draw Top Bar Background
            cv2.rectangle(frame, (0, 0), (1280, 80), (0, 0, 0), -1)
            
            # Draft Tokens (Yellow)
            draft_str = " ".join(draft_tokens)
            cv2.putText(frame, f"Draft: {draft_str}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Final Sentence (Green) - Swaps/Overlays
            if final_sentence:
                # Draw Box for sentence
                (w, h), _ = cv2.getTextSize(final_sentence, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                cv2.rectangle(frame, (20, 600), (20 + w + 20, 600 + h + 40), (0, 0, 0), -1)
                cv2.putText(frame, final_sentence, (30, 600 + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Debug info
            cv2.putText(frame, f"Live: {active_tk}", (1100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, status_text, (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow('Sign Inf (Smart)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        with open("debug_error.txt", "w") as f:
            f.write(traceback.format_exc())
        print("Error logged to debug_error.txt")
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
