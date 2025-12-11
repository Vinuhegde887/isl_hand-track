import numpy as np

def collapse_predictions(predictions, debounce_k=3, conf_min=0.5, merge_gap=0.6):
    """
    Refines raw per-frame predictions into stable tokens.
    
    Args:
        predictions: List of (token_str, confidence_float, timestamp_float).
                     Sorted by timestamp.
        debounce_k: Min consecutive frames to accept a token.
        conf_min: Min confidence to consider.
        merge_gap: Max seconds gap between identical tokens to merge them.
        
    Returns:
        List of dicts: [{'token': str, 'start': float, 'end': float, 'avg_conf': float}]
    """
    if not predictions:
        return []

    # 1. Filter by Confidence
    valid_preds = [p for p in predictions if p[1] >= conf_min]
    if not valid_preds:
        return []

    # 2. Debounce (Persistence Check)
    # Group consecutive identical tokens
    # We produce "candidate segments" which are sequences of identical tokens.
    
    candidates = []
    if not valid_preds: return []
    
    current_chain = [valid_preds[0]]
    
    for i in range(1, len(valid_preds)):
        curr = valid_preds[i]
        prev = valid_preds[i-1]
        
        # Check continuity (token match + time continuity approximately)
        # Assuming frames are dense. If there's a huge time gap, we consider it a break.
        time_gap = curr[2] - prev[2]
        
        if curr[0] == prev[0] and time_gap < 1.0: # 1 sec break breaks chain
            current_chain.append(curr)
        else:
            # Finalize chain
            if len(current_chain) >= debounce_k:
                candidates.append(current_chain)
            current_chain = [curr]
            
    # Last chain
    if len(current_chain) >= debounce_k:
        candidates.append(current_chain)

    # 3. Form collapsed events
    collapsed = []
    
    for chain in candidates:
        token = chain[0][0]
        start = chain[0][2]
        end = chain[-1][2]
        confs = [p[1] for p in chain]
        avg_conf = sum(confs) / len(confs)
        
        collapsed.append({
            'token': token,
            'start': start,
            'end': end,
            'avg_conf': avg_conf
        })

    # 4. Merge Gap (Merge identical tokens separated by short silence/noise)
    # "hello" (0.0-1.0) ... "hello" (1.4-2.0) -> Merge if gap < 0.6
    
    merged = []
    if not collapsed: return []
    
    current_event = collapsed[0]
    
    for i in range(1, len(collapsed)):
        next_event = collapsed[i]
        
        gap = next_event['start'] - current_event['end']
        
        if next_event['token'] == current_event['token'] and gap < merge_gap:
            # Merge
            # Update end time
            current_event['end'] = next_event['end']
            # Recompute avg conf (weighted by duration would be better, but simple avg is fine)
            current_event['avg_conf'] = (current_event['avg_conf'] + next_event['avg_conf']) / 2
        else:
            merged.append(current_event)
            current_event = next_event
            
    merged.append(current_event)
    
    return merged

def chunk_collapsed(collapsed, max_len=12, pause_threshold=0.8):
    """
    Groups collapsed events into sentence chunks.
    
    Args:
        collapsed: Output from collapse_predictions.
        max_len: Max tokens per chunk.
        pause_threshold: Seconds of silence between tokens to trigger chunk split.
        
    Returns:
        List of dicts: {tokens:[], confidences:[], timestamps:[], ...}
    """
    chunks = []
    if not collapsed: return []
    
    current_chunk = [collapsed[0]]
    
    for i in range(1, len(collapsed)):
        curr = collapsed[i]
        prev = collapsed[i-1]
        
        gap = curr['start'] - prev['end']
        
        # Split conditions
        is_long_pause = gap > pause_threshold
        is_too_long = len(current_chunk) >= max_len
        
        if is_long_pause or is_too_long:
            chunks.append(_finalize_chunk(current_chunk))
            current_chunk = [curr]
        else:
            current_chunk.append(curr)
            
    if current_chunk:
        chunks.append(_finalize_chunk(current_chunk))
        
    return chunks

def _finalize_chunk(event_list):
    tokens = [e['token'] for e in event_list]
    confs = [e['avg_conf'] for e in event_list]
    timestamps = [(e['start'], e['end']) for e in event_list]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    
    return {
        'tokens': tokens,
        'confidences': confs,
        'timestamps': timestamps,
        'avg_conf': avg_conf,
        'start': timestamps[0][0],
        'end': timestamps[-1][1]
    }
