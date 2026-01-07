# ===============================
# SYSTEM & PERFORMANCE FIXES
# ===============================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TF logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU

import sys
import json
import cv2
import numpy as np
import io
import requests
from tensorflow.lite import Interpreter

# UTF-8 FIX
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ===============================
# LABEL MAPS
# ===============================
EMOTION_MAP = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

AGE_MAP = {
    0: "Child",
    1: "Adult"
}

# ===============================
# LOAD MODELS ONCE (CRITICAL FIX)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.tflite")
AGE_MODEL_PATH = os.path.join(BASE_DIR, "child_adult_model.tflite")

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model missing: {path}")
    interpreter = Interpreter(model_path=path, num_threads=1)
    interpreter.allocate_tensors()
    return interpreter

try:
    EMOTION_INTERPRETER = load_model(EMOTION_MODEL_PATH)
    AGE_INTERPRETER = load_model(AGE_MODEL_PATH)

    EMOTION_IN = EMOTION_INTERPRETER.get_input_details()
    EMOTION_OUT = EMOTION_INTERPRETER.get_output_details()

    AGE_IN = AGE_INTERPRETER.get_input_details()
    AGE_OUT = AGE_INTERPRETER.get_output_details()

except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}))
    sys.exit(1)

# ===============================
# FACE DETECTOR (LOAD ONCE)
# ===============================
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# AGE PREDICTION
# ===============================
def predict_age(image_gray):
    faces = FACE_CASCADE.detectMultiScale(image_gray, 1.3, 5)
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = cv2.resize(image_gray[y:y+h, x:x+w], (48, 48))
    face = face.reshape(1, 48, 48, 1).astype("float32") / 255.0

    AGE_INTERPRETER.set_tensor(AGE_IN[0]["index"], face)
    AGE_INTERPRETER.invoke()
    output = AGE_INTERPRETER.get_tensor(AGE_OUT[0]["index"])

    if output.shape[-1] == 1:
        conf = float(output[0][0])
        return {
            "age_category": "Child" if conf < 0.5 else "Adult",
            "confidence": conf,
            "is_child": conf < 0.5
        }

    idx = int(np.argmax(output[0]))
    return {
        "age_category": AGE_MAP[idx],
        "confidence": float(output[0][idx]),
        "is_child": AGE_MAP[idx] == "Child"
    }

# ===============================
# EMOTION PREDICTION
# ===============================
def predict_emotion(image_gray):
    img = cv2.resize(image_gray, (48, 48))
    img = img.reshape(1, 48, 48, 1).astype("float32") / 255.0

    EMOTION_INTERPRETER.set_tensor(EMOTION_IN[0]["index"], img)
    EMOTION_INTERPRETER.invoke()
    output = EMOTION_INTERPRETER.get_tensor(EMOTION_OUT[0]["index"])[0]

    idx = int(np.argmax(output))
    return {
        "emotion": EMOTION_MAP[idx],
        "confidence": float(output[idx])
    }

# ===============================
# JAMENDO MUSIC
# ===============================
def get_music(emotion, limit=10):
    CLIENT_ID = "d8cdaaaf"
    TAGS = {
        "Happy": "happy",
        "Sad": "sad",
        "Angry": "rock",
        "Fear": "ambient",
        "Surprise": "dance",
        "Neutral": "chill"
    }.get(emotion, "instrumental")

    try:
        url = (
            f"https://api.jamendo.com/v3.0/tracks/"
            f"?client_id={CLIENT_ID}&format=json"
            f"&limit={limit}&tags={TAGS}"
        )
        r = requests.get(url, timeout=5)
        return [
            {
                "title": t["name"],
                "artist": t["artist_name"],
                "audio": t["audio"]
            }
            for t in r.json().get("results", [])
            if t.get("audio")
        ]
    except Exception:
        return []

# ===============================
# MAIN
# ===============================
def main():
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "error": "No image path"}))
        sys.exit(0)

    img_path = sys.argv[1]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(json.dumps({"status": "error", "error": "Invalid image"}))
        sys.exit(0)

    age = predict_age(img)
    emotion = predict_emotion(img)

    result = {
        "status": "success",
        "emotion": emotion["emotion"],
        "emotion_confidence": emotion["confidence"],
        "age_category": age["age_category"] if age else "Unknown",
        "music_recommendations": get_music(emotion["emotion"])
    }

    print(json.dumps(result))
    sys.stdout.flush()
    sys.exit(0)

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    main()
