"""
drive.py — Autonomous Driving Server
=====================================
This script connects to the Udacity Self-Driving Car Simulator using a
WebSocket (SocketIO) connection. When the simulator sends a camera image,
this script:
  1. Receives and decodes the image
  2. Pre-processes it (same as training: crop, resize, YUV, normalize)
  3. Feeds it into the trained NVIDIA CNN model
  4. Sends the predicted steering angle back to the simulator

Usage:
    1. Open the Udacity Simulator (Default Windows desktop 64-bit.exe)
    2. Choose a track and click "Autonomous Mode"
    3. Run this script: python drive.py best_model.h5
    4. Watch the car drive itself!

Requirements:
    pip install -r requirements.txt
"""

import argparse
import base64
import numpy as np
import cv2
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model

# ─── Configuration ───────────────────────────────────────────────────────────
IMG_HEIGHT   = 66       # NVIDIA CNN input height
IMG_WIDTH    = 200      # NVIDIA CNN input width
SPEED_LIMIT  = 20       # Maximum speed (mph)
# ─────────────────────────────────────────────────────────────────────────────

sio = socketio.Server()
app = Flask(__name__)
model = None   # Will be loaded from command-line argument


# ─── Image pre-processing (must match training exactly) ──────────────────────

def preprocess(img):
    """
    Pre-process raw camera image from the simulator so it matches
    the format used during training.
    
    Steps:
        1. Crop sky (top 60px) and car hood (bottom 25px)
        2. Resize to NVIDIA input: 66 x 200
        3. Convert BGR → YUV colorspace
    """
    img = img[60:-25, :, :]                         # 1. Crop
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))   # 2. Resize
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)       # 3. YUV
    return img


# ─── SocketIO Event Handlers ─────────────────────────────────────────────────

@sio.on('connect')
def connect(sid, environ):
    print(f'[+] Simulator connected (session id: {sid})')
    send_control(0, 0)   # Start with no movement


@sio.on('disconnect')
def disconnect(sid):
    print(f'[-] Simulator disconnected (session id: {sid})')


@sio.on('telemetry')
def telemetry(sid, data):
    """
    Called every time the simulator sends a new frame (telemetry packet).
    
    data contains:
        - image    : Base64-encoded JPEG image from center camera
        - speed    : Current speed of the car (mph)
        - throttle : Current throttle value
        - steering_angle : Current steering angle
    """
    if data is None:
        send_control(0, 0)
        return

    # ── 1. Decode the image ──────────────────────────────────────────────────
    img_b64  = data['image']
    img_bytes = base64.b64decode(img_b64)
    img_pil   = Image.open(BytesIO(img_bytes))
    img_np    = np.array(img_pil)                    # Shape: (H, W, 3) RGB

    # ── 2. Pre-process ───────────────────────────────────────────────────────
    img_processed = preprocess(img_np)
    img_normalized = img_processed.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)  # Shape: (1, 66, 200, 3)

    # ── 3. Predict steering angle ────────────────────────────────────────────
    steering_angle = float(model.predict(img_input, verbose=0)[0][0])

    # ── 4. Calculate throttle (slow down on sharp turns) ─────────────────────
    current_speed = float(data['speed'])
    if current_speed < SPEED_LIMIT:
        throttle = 1.0 - (abs(steering_angle) * 2.0)  # reduce throttle on turns
        throttle = max(0.1, min(throttle, 1.0))        # clamp between 0.1 and 1.0
    else:
        throttle = 0.0   # coast if at speed limit

    print(f'Steering: {steering_angle:+.4f} | Throttle: {throttle:.2f} | Speed: {current_speed:.1f} mph')

    # ── 5. Send control back to simulator ────────────────────────────────────
    send_control(steering_angle, throttle)


def send_control(steering_angle, throttle):
    """Send steering angle and throttle commands to the simulator."""
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle'      : str(throttle)
    })


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autonomous Driving Server')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default='best_model.h5',
        help='Path to the trained Keras model file (default: best_model.h5)'
    )
    args = parser.parse_args()

    print(f'[*] Loading model: {args.model}')
    model = load_model(args.model)
    print(f'[*] Model loaded successfully!')
    print(f'[*] Starting autonomous driving server on port 4567...')
    print(f'[*] Open the simulator and select "Autonomous Mode"')
    print()

    # Wrap Flask + SocketIO app with eventlet WSGI server
    app = socketio.WSGIApp(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
