import torch
import numpy as np
from collections import deque
import logging

class InferenceEmitter:
    def __init__(self, model, label_map, fps=30, window_size=30, stride=1, device=None, debounce_k=3, conf_min=0.5):
        """
        Args:
            model: PyTorch model (eval mode).
            label_map: List of class names or dict {index: name}.
            fps: Frames per second of input stream.
            window_size: Number of frames required for one inference (e.g. 30).
            stride: How many frames to slide before next inference (default 1 = every frame).
            device: torch.device.
            debounce_k: Number of consecutive frames with same prediction to trigger emission.
            conf_min: Minimum confidence to consider a prediction valid.
        """
        self.model = model
        self.label_map = label_map
        self.fps = fps
        self.window_size = window_size
        self.stride = stride
        self.device = device or torch.device('cpu')
        self.debounce_k = debounce_k
        self.conf_min = conf_min

        self.model.eval()
        self.model.to(self.device)

        # Buffers
        self.frame_buffer = deque(maxlen=window_size)
        self.stride_counter = 0

        # State Machine for Emission
        self.current_token = None
        self.current_start_time = None
        self.current_end_time = None
        self.current_confs = []
        
        # Debounce State
        self.pending_token = None
        self.pending_count = 0
        self.pending_confs = []
        self.pending_start_time = None

        self.emitted_events = []
        
        # Diagnostics
        self.logger = logging.getLogger(__name__)

    def reset(self):
        self.frame_buffer.clear()
        self.stride_counter = 0
        self.current_token = None
        self.current_start_time = None
        self.current_end_time = None
        self.current_confs = []
        self.pending_token = None
        self.pending_count = 0
        self.emitted_events = []

    def get_emitted_events(self):
        return self.emitted_events

    def process_frame(self, landmark_vec, timestamp):
        """
        Args:
            landmark_vec: List or array of features for one frame.
            timestamp: Float timestamp (seconds).
        Returns:
            Dict of emitted token event if a token just completed/changed, else None.
        """
        self.frame_buffer.append(landmark_vec)
        
        # Only classify when buffer fills up
        if len(self.frame_buffer) < self.window_size:
            return None

        # Stride Control
        self.stride_counter += 1
        if self.stride_counter < self.stride:
            return None
        self.stride_counter = 0

        # Prepare Input
        input_tensor = torch.tensor(list(self.frame_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            conf = conf.item()
            idx = idx.item()
            
        token = self.label_map[idx] if isinstance(self.label_map, list) else self.label_map.get(idx, "Unknown")
        
        # Background Suppression (assuming 'nothing' or 'background' is a class)
        if token in ['nothing', 'background'] or conf < self.conf_min:
            self._handle_silence(timestamp)
            return self._check_emission(force_close=False)

        # Debounce Logic
        if token == self.pending_token:
            self.pending_count += 1
            self.pending_confs.append(conf)
        else:
            # Reset pending if token changes
            self.pending_token = token
            self.pending_count = 1
            self.pending_confs = [conf]
            self.pending_start_time = timestamp # Roughly start of new token detection

        # Check if pending becomes stable
        if self.pending_count >= self.debounce_k:
            return self._update_active_token(self.pending_token, self.pending_start_time, timestamp, self.pending_confs[-1])
        
        return None

    def _update_active_token(self, token, start_time, end_time, conf):
        """
        Called when we have a stable token detection.
        Decides whether to extend the current event or emit the old one and start new.
        """
        emitted_event = None

        if self.current_token == token:
            # Extend current
            self.current_end_time = end_time
            self.current_confs.append(conf)
        else:
            # Switch happening
            if self.current_token is not None:
                # Emit Previous
                emitted_event = self._finalize_event()
            
            # Start New
            self.current_token = token
            self.current_start_time = start_time
            self.current_end_time = end_time
            self.current_confs = [conf]

        return emitted_event

    def _handle_silence(self, timestamp):
        """
        If we see silence/background, we might want to close the current event if enough time passes.
        Here we reset pending.
        """
        self.pending_token = None
        self.pending_count = 0
        self.pending_confs = []
        
        # Optional: Auto-close current token if silence persists?
        # For now, we only close if a *new* different token appears, 
        # but we could add a timeout here.
        pass

    def _finalize_event(self):
        """Package the current token into an event dict."""
        if self.current_token is None:
            return None
        
        avg_conf = sum(self.current_confs) / max(1, len(self.current_confs))
        event = {
            'token': self.current_token,
            'start_time': self.current_start_time,
            'end_time': self.current_end_time,
            'avg_conf': avg_conf,
            'count': len(self.current_confs)
        }
        self.emitted_events.append(event)
        
        # Reset current
        self.current_token = None
        self.current_start_time = None
        self.current_end_time = None
        self.current_confs = []
        return event

    def _check_emission(self, force_close=False):
        """Helper to force close if needed (e.g. end of stream)."""
        if force_close and self.current_token:
            return self._finalize_event()
        return None
