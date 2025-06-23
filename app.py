from flask import Flask, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import hashlib
import random
import math
import urllib.request
import os

app = Flask(__name__)
CORS(app)

# Video source
VIDEO_URL = "https://antropy-api.onrender.com/ants_compressed.mp4"
LOCAL_VIDEO = "video.mp4"

# Configuration
GRID_SIZE = (2, 2)
THRESHOLD_DELTA = 6.4

def compute_entropy(bitstring):
    counts = [bitstring.count('0'), bitstring.count('1')]
    total = sum(counts)
    probs = [c / total for c in counts if c > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    return entropy

@app.route('/', methods=['GET'])
def index():
    return "<h3>Antropy Hash API is running.<br>Use <code>/generate</code> to get a hash.</h3>"

@app.route('/generate', methods=['GET'])
def generate():
    try:
        urllib.request.urlretrieve(VIDEO_URL, LOCAL_VIDEO)
    except Exception as e:
        return jsonify({"error": "Failed to download video", "details": str(e)}), 500

    cap = cv2.VideoCapture(LOCAL_VIDEO)
    if not cap.isOpened():
        return jsonify({"error": "Failed to open video stream"}), 500

    activity_bits = []
    prev_gray = None
    frame_count = 0
    skip = random.randint(5, 50)
    for _ in range(skip):
        cap.read()

    while len(activity_bits) < 256 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mask = (diff > THRESHOLD_DELTA).astype(np.uint8)

            h, w = mask.shape
            gh, gw = h // GRID_SIZE[0], w // GRID_SIZE[1]

            for y in range(GRID_SIZE[0]):
                for x in range(GRID_SIZE[1]):
                    region = mask[y*gh:(y+1)*gh, x*gw:(x+1)*gw]
                    changed_ratio = np.mean(region)
                    if changed_ratio < 0.02:
                        bits = "00"
                    elif changed_ratio < 0.06:
                        bits = "01"
                    elif changed_ratio < 0.12:
                        bits = "10"
                    else:
                        bits = "11"
                    activity_bits.extend([int(b) for b in bits])

        prev_gray = gray
        frame_count += 1

        for _ in range(random.randint(1, 4)):
            cap.read()

    cap.release()

    print(f"Frames captured: {frame_count}, Bits collected: {len(activity_bits)}")

    if len(activity_bits) < 256:
        return jsonify({"error": "Not enough data collected to generate hash"}), 500

    binary_string = ''.join(map(str, activity_bits[:256]))
    bit_bytes = int(binary_string, 2).to_bytes(32, byteorder='big')
    hash_digest = hashlib.sha256(bit_bytes).hexdigest()
    entropy = compute_entropy(binary_string)

    try:
        os.remove(LOCAL_VIDEO)
    except:
        pass

    return jsonify({
        "binary": binary_string,
        "hash": hash_digest,
        "entropy": round(entropy, 5)
    })

if __name__ == '__main__':
    app.run(debug=True)
