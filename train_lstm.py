import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Configuration
CSV_PATH = r"d:/isl updated/keypoints_data.csv"
MODEL_SAVE_PATH = r"d:/isl updated/action.pth" # Changed extension to .pth
SEQUENCE_LENGTH = 60
FEATURES_PER_FRAME = 126
HIDDEN_SIZE = 64
NUM_LAYERS = 4
NUM_EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class HandGestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Added dropout to LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        
        # Simplified FC layers and added Dropout
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes) 
        # Removed one extra layer to reduce complexity

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        
        out = self.dropout(out) # Apply dropout
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out) # Apply dropout
        out = self.fc2(out)
        return out

def load_and_preprocess_data():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file '{CSV_PATH}' not found. Please run preprocess_lstm.py first.")
        return None, None, None

    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    
    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except last
    y = df.iloc[:, -1].values   # Last column (label)
    
    # Reshape X to (samples, 50, 126)
    num_samples = X.shape[0]
    X = X.reshape(num_samples, SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    classes = label_encoder.classes_
    print(f"Classes: {classes}")
    
    return X, y_encoded, classes

def main():
    X, y, classes = load_and_preprocess_data()
    if X is None:
        return

    # Split data - Added stratify to ensure balanced split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create Datasets and DataLoaders
    train_dataset = HandGestureDataset(X_train, y_train)
    test_dataset = HandGestureDataset(X_test, y_test)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    num_classes = len(classes)
    model = LSTMModel(FEATURES_PER_FRAME, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Added weight_decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Train
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total

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
        
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Final Evaluation (Optional, since we print every epoch)
    print(f'Final Test Accuracy: {val_acc:.2f} %')
    
    # Save
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
