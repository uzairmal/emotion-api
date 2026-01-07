import os
import sys
import json
import cv2
import numpy as np
import io
import requests
from tflite_runtime.interpreter import Interpreter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMOTION_MODEL = os.path.join(BASE_DIR, "emotion_model.tflite")
AGE_MODEL = os.path.join(BASE_DIR, "child_adult_model.tflite")

EMOTION_MAP = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

FACE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

emotion_interpreter = Interpreter(EMOTION_MODEL)
age_interpreter = Interpreter(AGE_MODEL)

emotion_interpreter.allocate_tensors()
age_interpreter.allocate_tensors()

EI = emotion_interpreter.get_input_details()
EO = emotion_interpreter.get_output_details()
AI = age_interpreter.get_input_details()
AO = age_interpreter.get_output_details()

def get_music(emotion):
    try:
        url = f"https://api.jamendo.com/v3.0/tracks/?client_id=d8cdaaaf&limit=5&tags={emotion.lower()}"
        r = requests.get(url, timeout=4)
        return [
            {"title": t["name"], "artist": t["artist_name"], "audio": t["audio"]}
            for t in r.json().get("results", [])
        ]
    except:
        return []

def main():
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(json.dumps({"status": "error"}))
        return

    faces = FACE.detectMultiScale(img, 1.3, 5)
    age = "Unknown"

    if len(faces):
        x,y,w,h = faces[0]
        f = cv2.resize(img[y:y+h, x:x+w], (48,48))
        f = f.reshape(1,48,48,1).astype("float32")/255
        age_interpreter.set_tensor(AI[0]["index"], f)
        age_interpreter.invoke()
        age = "Child" if age_interpreter.get_tensor(AO[0]["index"])[0][0] < 0.5 else "Adult"

    img48 = cv2.resize(img, (48,48))
    img48 = img48.reshape(1,48,48,1).astype("float32")/255
    emotion_interpreter.set_tensor(EI[0]["index"], img48)
    emotion_interpreter.invoke()
    out = emotion_interpreter.get_tensor(EO[0]["index"])[0]
    emo = EMOTION_MAP[int(np.argmax(out))]

    print(json.dumps({
        "status": "success",
        "emotion": emo,
        "age": age,
        "music": get_music(emo)
    }))

if __name__ == "__main__":
    main()
