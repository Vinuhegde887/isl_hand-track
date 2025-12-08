#!/usr/bin/env python3
"""
Live Picamera2 preview with MediaPipe hand-landmark overlay.
Press 'q' to quit.
"""

from picamera2 import Picamera2
import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# ---------- MediaPipe init ----------
mp_drawing   = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands     = mp.solutions.hands
hands        = mp_hands.Hands(static_image_mode=False,
                              max_num_hands=2,
                              min_detection_confidence=0.7,
                              min_tracking_confidence=0.5)

# ---------- Picamera2 init ----------
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (960, 720), "format": "RGB888"})
picam2.configure(config)
picam2.start()

GPIO.setmode(GPIO.BCM)     # Use BCM pin numbering
GPIO.setwarnings(False)
PIN = 26                   # GPIO 26

GPIO.setup(PIN, GPIO.OUT)  # Set pin as OUTPUT

GPIO.output(PIN, GPIO.HIGH)  # Set HIGH
print("GPIO 26 is now HIGH")

print("Hand-landmark preview (q to quit)")
while True:
    frame_rgb = picam2.capture_array()  
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
             # RGB
    results   = hands.process(frame_bgr)             # landmarks

    # convert to BGR for OpenCV drawing
    frame_bgr = frame_rgb

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow("MediaPipe Hands", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- cleanup ----------
hands.close()
picam2.stop()
cv2.destroyAllWindows()
