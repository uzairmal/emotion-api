import sys
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.lite import Interpreter
import os
import requests
import io

# ===============================
# UTF-8 FIX (Render / Windows)
# ===============================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ===============================
# LABEL MAPS
# ===============================
emotion_map = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

age_map = {
    0: "Child",
    1: "Adult"
}

# ===============================
# MODEL LOADER
# ===============================
def load_tflite_model(model_path):
    try:
        if not os.path.exists(model_path):
            return None, f"Model file not found: {model_path}"

        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        return {
            "interpreter": interpreter,
            "input_details": interpreter.get_input_details(),
            "output_details": interpreter.get_output_details()
        }, None

    except Exception as e:
        return None, str(e)

# ===============================
# AGE PREDICTION
# ===============================
def predict_age(image_path, model_path="child_adult_model.tflite"):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, "Failed to load image"

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        if len(faces) == 0:
            return None, "No face detected"

        x, y, w, h = faces[0]
        face = cv2.resize(img[y:y+h, x:x+w], (48, 48))
        face = face.reshape(1, 48, 48, 1).astype("float32") / 255.0

        model, err = load_tflite_model(model_path)
        if err:
            return None, err

        interp = model["interpreter"]
        interp.set_tensor(model["input_details"][0]["index"], face)
        interp.invoke()
        output = interp.get_tensor(model["output_details"][0]["index"])

        if output.shape[-1] == 1:
            confidence = float(output[0][0])
            is_child = confidence < 0.5
            age = "Child" if is_child else "Adult"
        else:
            idx = int(np.argmax(output[0]))
            confidence = float(output[0][idx])
            age = age_map.get(idx, "Unknown")
            is_child = age == "Child"

        return {
            "age_category": age,
            "confidence": confidence,
            "is_child": is_child
        }, None

    except Exception as e:
        return None, str(e)

# ===============================
# EMOTION PREDICTION
# ===============================
def predict_emotion(image_path, model_path="emotion_model.tflite"):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, "Failed to load image"

        img = cv2.resize(img, (48, 48))
        img = img.reshape(1, 48, 48, 1).astype("float32") / 255.0

        model, err = load_tflite_model(model_path)
        if err:
            return None, err

        interp = model["interpreter"]
        interp.set_tensor(model["input_details"][0]["index"], img)
        interp.invoke()
        output = interp.get_tensor(model["output_details"][0]["index"])[0]

        idx = int(np.argmax(output))
        emotion = emotion_map.get(idx, "Unknown")

        return {
            "emotion": emotion,
            "confidence": float(output[idx]),
            "all_probabilities": {
                emotion_map[i]: float(output[i])
                for i in range(len(emotion_map))
            }
        }, None

    except Exception as e:
        return None, str(e)

# ===============================
# JAMENDO MUSIC
# ===============================
def get_music_recommendations(emotion, is_child, limit=10):
    CLIENT_ID = "d8cdaaaf"

    tag_map = {
        "Happy": "happy,upbeat",
        "Sad": "sad,acoustic",
        "Angry": "rock,energetic",
        "Fear": "ambient",
        "Surprise": "dance",
        "Neutral": "chill"
    }

    tags = tag_map.get(emotion, "instrumental")

    try:
        url = (
            f"https://api.jamendo.com/v3.0/tracks/"
            f"?client_id={CLIENT_ID}&format=json"
            f"&limit={limit}&tags={tags}&audioformat=mp31"
        )

        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []

        data = r.json().get("results", [])
        songs = []

        for t in data:
            if t.get("audio"):
                songs.append({
                    "id": str(t["id"]),
                    "title": t["name"],
                    "artist": t["artist_name"],
                    "audio": t["audio"],
                    "image": t.get("album_image", "")
                })

        return songs[:limit]

    except Exception:
        return []

# ===============================
# MAIN ENTRY POINT
# ===============================
def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"status": "error", "error": "No image path provided"}))
            return

        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(json.dumps({"status": "error", "error": "Image not found"}))
            return

        age_result, age_error = predict_age(image_path)
        emotion_result, emotion_error = predict_emotion(image_path)

        if age_error and emotion_error:
            result = {
                "status": "error",
                "age_error": age_error,
                "emotion_error": emotion_error
            }

        elif emotion_error:
            result = {
                "status": "error",
                "emotion_error": emotion_error,
                "age_result": age_result
            }

        elif age_error:
            result = {
                "status": "error",
                "age_error": age_error,
                "emotion": emotion_result["emotion"]
            }

        else:
            songs = get_music_recommendations(
                emotion_result["emotion"],
                age_result["is_child"]
            )

            result = {
                "status": "success",
                "age_category": age_result["age_category"],
                "age_confidence": age_result["confidence"],
                "emotion": emotion_result["emotion"],
                "emotion_confidence": emotion_result["confidence"],
                "music_recommendations": songs
            }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    main()
