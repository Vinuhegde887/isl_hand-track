import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import math
import os

# Configuration
CSV_PATH = "keypoints_data.csv"
MODEL_SAVE_PATH = "transformer_action.pth"
SEQUENCE_LENGTH = 30 # Must match preprocess
FEATURES_PER_FRAME = 126
NUM_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.0001 # Transformers usually need lower LR
NUM_CLASSES = 0 # Will be set dynamically

# Transformer Hyperparameters
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
        
        # Input embedding layer to project features to d_model size
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=SEQUENCE_LENGTH)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        
        # Project to d_model
        src = self.embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        # Permute for Positional Encoding which expects [seq_len, batch, d_model] if batch_first=False
        # But we used batch_first=True in TransformerEncoder, so we can keep it, 
        # BUT our PositionalEncoding helper is written for [seq_len, batch, d_model].
        # Let's adjust PositionalEncoding usage.
        
        src = src.permute(1, 0, 2) # [seq_len, batch, d_model]
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2) # Back to [batch, seq_len, d_model]
        
        output = self.transformer_encoder(src)
        
        # Global Average Pooling
        output = output.mean(dim=1) 
        
        output = self.dropout(output)
        output = self.fc_out(output)
        return output

class HandGestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file '{CSV_PATH}' not found.")
        return None, None, None

    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    
    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    try:
        X = X.reshape(X.shape[0], SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    except ValueError as e:
        print(f"Error reshaping data: {e}. Check SEQUENCE_LENGTH.")
        return None, None, None
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    
    return X, y_encoded, classes

def main():
    X, y, classes = load_data()
    if X is None:
        return

    num_classes = len(classes)
    print(f"Classes: {classes}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = HandGestureDataset(X_train, y_train)
    test_dataset = HandGestureDataset(X_test, y_test)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SignLanguageTransformer(
        num_classes=num_classes, 
        input_dim=FEATURES_PER_FRAME,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Added weight decay
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / val_total
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved.")

if __name__ == "__main__":
    main()
