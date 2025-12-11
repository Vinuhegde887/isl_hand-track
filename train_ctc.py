import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_synthetic import VideoLoader, SyntheticContinuousSignDataset, synthetic_collate_fn
import os
import math

# Configuration
DATA_DIR = "data"
MODEL_SAVE_PATH = "transformer_ctc_refined.pth"
NUM_EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
FEATURES_PER_FRAME = 126

# Model Hyperparameters
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 3 # Increased to 3 as suggested
DIM_FEEDFORWARD = 256
DROPOUT = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [1, MaxLen, D] -> [MaxLen, 1, D] usually? 
        # Actually standard is [MaxLen, 1, D] or [1, MaxLen, D].
        # Let's stick to adding to [B, T, D]
        pe = pe.permute(1, 0, 2) # [1, MaxLen, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]

class SignCTCModel(nn.Module):
    def __init__(self, vocab_size, input_dim=126, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.5):
        super(SignCTCModel, self).__init__()
        
        self.d_model = d_model
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
        # Output: vocab_size (6 classes) + 1 blank = 7 logits
        # blank index will be 6
        self.fc_out = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x, src_key_padding_mask=None):
        # x: [B, T, 126]
        # src_key_padding_mask: [B, T] (True where padded)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # TransformerEncoder takes src_key_padding_mask
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        output = self.dropout(output)
        output = self.fc_out(output) # [B, T, V+1]
        
        # Return [T, B, V+1] for CTCLoss
        return output.permute(1, 0, 2)

def main():
    if not os.path.exists(DATA_DIR):
        print("Data directory not found.")
        return

    print("Initializing Data Loader...")
    loader = VideoLoader(DATA_DIR)
    clips, classes = loader.load_all_clips()
    print(f"Loaded {len(clips)} clips.")
    vocab_size = len(classes)
    print(f"Vocab Size: {vocab_size} (Classes: {classes})")
    blank_idx = vocab_size # Index 6 if 6 classes (0-5)
    
    train_dataset = SyntheticContinuousSignDataset(clips, num_samples=1000, min_seq=2, max_seq=5)
    val_dataset = SyntheticContinuousSignDataset(clips, num_samples=100, min_seq=2, max_seq=5)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=synthetic_collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=synthetic_collate_fn, num_workers=2)
    
    model = SignCTCModel(
        vocab_size=vocab_size,
        input_dim=FEATURES_PER_FRAME,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    # CTCLoss setup
    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    print(f"Starting Training (Blank Index: {blank_idx})...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_targets, input_lengths, target_lengths in train_loader:
            batch_x = batch_x.to(device)
            batch_targets = batch_targets.to(device)
            
            # Create padding mask for Transformer
            # Mask is [B, T], True where padded (index >= length)
            # We can create it from input_lengths
            max_len = batch_x.size(1)
            # arange: [0, 1, ... max_len-1]
            # unsqueeze(0): [1, max_len]
            # input_lengths.unsqueeze(1): [B, 1]
            key_padding_mask = torch.arange(max_len).to(device).unsqueeze(0) >= input_lengths.to(device).unsqueeze(1)
            
            # Forward
            # output: [T, B, V+1]
            logits = model(batch_x, src_key_padding_mask=key_padding_mask)
            
            # Log Softmax for CTC
            log_probs = logits.log_softmax(2)
            
            loss = criterion(log_probs, batch_targets, input_lengths, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_targets, input_lengths, target_lengths in val_loader:
                batch_x = batch_x.to(device)
                batch_targets = batch_targets.to(device)
                max_len = batch_x.size(1)
                key_padding_mask = torch.arange(max_len).to(device).unsqueeze(0) >= input_lengths.to(device).unsqueeze(1)
                
                logits = model(batch_x, src_key_padding_mask=key_padding_mask)
                log_probs = logits.log_softmax(2)
                loss = criterion(log_probs, batch_targets, input_lengths, target_lengths)
                val_loss += loss.item()
                
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] CTC Loss: {avg_loss:.4f} | Val: {avg_val:.4f}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"checkpoint saved to {MODEL_SAVE_PATH}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved.")

if __name__ == "__main__":
    main()
