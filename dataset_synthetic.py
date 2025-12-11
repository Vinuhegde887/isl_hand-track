import os
import cv2
import torch
import numpy as np
import random
import mediapipe as mp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Configuration (Reuse existing constants)
FEATURES_PER_FRAME = 126
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3

class VideoLoader:
    def __init__(self, data_dir, method='mediapipe'):
        self.data_dir = data_dir
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=NUM_HANDS,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) if method == 'mediapipe' else None

    def extract_features_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames_keypoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Mediapipe extraction
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(image_rgb)
            
            frame_kps = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        frame_kps.extend([lm.x, lm.y, lm.z])
                # Pad if missing hands
                detected = len(results.multi_hand_landmarks)
                if detected < NUM_HANDS:
                    frame_kps.extend([0.0] * (NUM_HANDS - detected) * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK)
            else:
                frame_kps.extend([0.0] * FEATURES_PER_FRAME)
            
            frames_keypoints.append(frame_kps[:FEATURES_PER_FRAME])
        
        cap.release()
        return np.array(frames_keypoints, dtype=np.float32)

    def load_all_clips(self):
        """Returns a list of dicts: {'features': np_array, 'label': int_idx}"""
        import pickle
        
        cache_path = os.path.join(self.data_dir, "features_cache.pkl")
        cached_data = {}
        
        # Load Cache if exists
        if os.path.exists(cache_path):
            print(f"Loading features from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}. Will re-process.")
                cached_data = {}
        
        all_clips = []
        cache_updated = False
        print(f"Scanning clips from {self.data_dir}...")
        
        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            cls_path = os.path.join(self.data_dir, cls_name)
            video_files = [f for f in os.listdir(cls_path) if f.endswith(('.mp4', '.avi'))]
            
            print(f"  Class {cls_name}: {len(video_files)} videos")
            for vid in video_files:
                rel_path = os.path.join(cls_name, vid) # Unique key
                path = os.path.join(cls_path, vid)
                
                # Check cache
                if rel_path in cached_data:
                    features = cached_data[rel_path]
                else:
                    # Extract fresh
                    print(f"    New/Updated video: {vid}")
                    features = self.extract_features_from_video(path)
                    cached_data[rel_path] = features
                    cache_updated = True
                
                if len(features) > 0:
                    all_clips.append({'features': features, 'label': cls_idx})
        
        # Save cache if changed
        if cache_updated:
            print("Saving updated cache...")
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
                    
        return all_clips, self.classes

class SyntheticContinuousSignDataset(Dataset):
    def __init__(self, clips, num_samples=1000, min_seq=2, max_seq=5):
        """
        clips: List of cached {'features', 'label'} loaded from VideoLoader
        num_samples: How many synthetic sequences to generate per epoch
        """
        self.clips = clips
        self.num_samples = num_samples
        self.min_seq = min_seq
        self.max_seq = max_seq

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Randomly decide how many clips to concat
        num_clips = random.randint(self.min_seq, self.max_seq)
        
        # 2. Sample random clips
        chosen_indices = random.choices(range(len(self.clips)), k=num_clips)
        chosen_clips = [self.clips[i] for i in chosen_indices]
        
        # 3. Concatenate features along time axis
        # List of [T, 126] -> Single [Total_T, 126]
        features_list = [c['features'] for c in chosen_clips]
        
        # Optional: Add "silence" or transition frames between clips?
        # For now, pure concatenation as requested.
        X = np.concatenate(features_list, axis=0)
        
        # 4. Build target list
        labels_list = [c['label'] for c in chosen_clips]
        y = np.array(labels_list, dtype=np.int64)
        
        return torch.from_numpy(X), torch.from_numpy(y)

def synthetic_collate_fn(batch):
    """
    batch: List of tuples (X, y)
    Returns:
       batch_x: [B, T_max, 126] (Padded)
       batch_targets: [Total_Targets] (Concatenated 1D)
       input_lengths: [B]
       target_lengths: [B]
    """
    # Separate input and targets
    inputs = [item[0] for item in batch] # List of Tensors
    targets = [item[1] for item in batch] # List of Tensors
    
    # 1. Input Lengths
    input_lengths = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)
    
    # 2. Pad Inputs (batch_first=True -> [B, T_max, Features])
    batch_x = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    
    # 3. Target Lengths
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    
    # 4. Concatenate Targets (CTC loss expects a single 1D vector of all targets)
    batch_targets = torch.cat(targets, dim=0)
    
    return batch_x, batch_targets, input_lengths, target_lengths

if __name__ == "__main__":
    # Test script
    DATA_DIR = "data"
    if os.path.exists(DATA_DIR):
        print("Initializing Loader...")
        loader = VideoLoader(DATA_DIR)
        clips, classes = loader.load_all_clips()
        print(f"Loaded {len(clips)} total isolated clips.")
        
        # Create Dataset
        dataset = SyntheticContinuousSignDataset(clips, num_samples=10)
        
        # Manually invoke getitem
        X, y = dataset[0]
        print(f"Sample 0 Shapes -- X: {X.shape}, y: {y.shape}")
        print(f"Sample 0 Targets: {[classes[i] for i in y]}")
        
        # Test DataLoader
        from torch.utils.data import DataLoader
        dl = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate_fn)
        
        batch = next(iter(dl))
        batch_x, batch_y, input_lens, target_lens = batch
        print("\nBatch Stats:")
        print(f"Batch X Shape: {batch_x.shape}")
        print(f"Batch Targets Shape: {batch_y.shape}")
        print(f"Input Lengths: {input_lens}")
        print(f"Target Lengths: {target_lens}")
    else:
        print("Data directory not found, skipping test.")
