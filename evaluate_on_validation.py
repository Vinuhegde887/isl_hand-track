import torch
import numpy as np
import random
import os
from itertools import groupby

# Import from existing project files
try:
    from train_ctc import SignCTCModel
    from dataset_synthetic import VideoLoader, SyntheticContinuousSignDataset
except ImportError:
    # Fallback if running from a different directory structure
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train_ctc import SignCTCModel
    from dataset_synthetic import VideoLoader, SyntheticContinuousSignDataset

# Settings
DATA_DIR = "data"
MODEL_PATH = "transformer_ctc_refined.pth"
NUM_VALIDATION_SAMPLES = 50
RANDOM_SEED = 42

FEATURES_PER_FRAME = 126
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def levenshtein_distance(ref, hyp):
    """
    Computes Levenshtein distance between two sequences.
    """
    m, n = len(ref), len(hyp)
    if m < n:
        return levenshtein_distance(hyp, ref)
    if n == 0:
        return m

    previous_row = range(n + 1)
    for i, c1 in enumerate(ref):
        current_row = [i + 1]
        for j, c2 in enumerate(hyp):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def greedy_decode(logits, classes, blank_idx):
    """
    Decodes CTC logits using greedy strategy.
    logits: [SeqLen, B, VocabSize] or [SeqLen, VocabSize]
    """
    if logits.dim() == 3:
        logits = logits.squeeze(1) # Assume B=1
        
    probs = torch.softmax(logits, dim=-1)
    max_probs, indices = torch.max(probs, dim=-1)
    indices = indices.cpu().numpy()
    
    decoded_inds = []
    for i, (k, g) in enumerate(groupby(indices)):
        if k != blank_idx:
            decoded_inds.append(k)
            
    # Convert to tokens
    result = [classes[i] for i in decoded_inds]
    return result

def main():
    # 1. Set Deterministic Seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found.")
        return

    print("Loading Data...")
    loader = VideoLoader(DATA_DIR)
    # The loader caching might affect things, but the content is static.
    all_clips, all_classes = loader.load_all_clips()

    # FILTER: RESTRICT TO TRAINED CLASSES TO MATCH CHECKPOINT
    TRAINED_CLASSES = ['a', 'hello', 'm', 'name', 'nothing', 'person', 'what']
    print(f"Filtering dataset to match trained model classes: {TRAINED_CLASSES}")
    
    # Create mapping from old index to new index (and filter out unused)
    # old_idx -> class_name -> new_idx
    old_to_new_label = {}
    for i, cls in enumerate(all_classes):
        if cls in TRAINED_CLASSES:
            old_to_new_label[i] = TRAINED_CLASSES.index(cls)
            
    filtered_clips = []
    for clip in all_clips:
        old_label = clip['label']
        if old_label in old_to_new_label:
            # Create a shallow copy to modify label without breaking cache if needed (though here it's fine)
            new_clip = clip.copy()
            new_clip['label'] = old_to_new_label[old_label]
            filtered_clips.append(new_clip)
            
    clips = filtered_clips
    classes = TRAINED_CLASSES
    vocab_size = len(classes)
    blank_idx = vocab_size # 7
    print(f"Filtered Vocab: {classes}, Blank: {blank_idx}, Valid Clips: {len(clips)}")

    # 2. Create Fixed Validation List
    # We use the dataset class but we'll access it sequentially after seeding
    val_dataset = SyntheticContinuousSignDataset(clips, num_samples=NUM_VALIDATION_SAMPLES, min_seq=3, max_seq=6)
    
    print("Loading Model...")
    model = SignCTCModel(
        vocab_size=vocab_size, # Should be 7
        input_dim=FEATURES_PER_FRAME,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Model file {MODEL_PATH} not found.")
        return
        
    model.eval()
    
    total_dist = 0
    total_len = 0
    exact_matches = 0
    
    print(f"\nEvaluating on {NUM_VALIDATION_SAMPLES} fixed samples...")
    print(f"{'Ground Truth':<40} | {'Prediction':<40} | {'Dist'}")
    print("-" * 90)
    
    # Iterate through the fixed dataset
    for i in range(NUM_VALIDATION_SAMPLES):
        # Since we seeded `random` at start, this __getitem__ logic should be deterministic 
        # as long as Synthetic ContinuousSignDataset uses the global `random` module 
        # (which it defines at the top level).
        X, y = val_dataset[i] 
        
        # Ground Truth
        gt_tokens = [classes[idx] for idx in y]
        gt_str = " ".join(gt_tokens)
        
        # Inference
        input_tensor = X.unsqueeze(0).to(DEVICE) # [1, T, 126]
        
        with torch.no_grad():
            output = model(input_tensor) # [T, 1, V+1] from train_ctc.py logic
            
        pred_tokens = greedy_decode(output, classes, blank_idx)
        pred_str = " ".join(pred_tokens)
        
        # Metrics
        dist = levenshtein_distance(gt_tokens, pred_tokens)
        total_dist += dist
        total_len += len(gt_tokens)
        
        if gt_tokens == pred_tokens:
            exact_matches += 1
            
        print(f"{gt_str:<40} | {pred_str:<40} | {dist}")

    print("-" * 90)
    wer = total_dist / total_len if total_len > 0 else 0
    accuracy = exact_matches / NUM_VALIDATION_SAMPLES
    
    print(f"Final Validation Results (Seed={RANDOM_SEED}):")
    print(f"Samples: {NUM_VALIDATION_SAMPLES}")
    print(f"Exact Match Accuracy: {accuracy:.2%}")
    print(f"Avg Edit Distance (WER): {wer:.2%}")

if __name__ == "__main__":
    main()
