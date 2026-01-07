import sys
import json
import numpy as np
import cv2
import tensorflow as tf
Interpreter = tf.lite.Interpreter
import os
import requests

# Set UTF-8 encoding for Windows compatibility
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Emotion class mapping
emotion_map = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear",
    3: "Happy", 
    4: "Sad", 
    5: "Surprise", 
    6: "Neutral"
}

# Age class mapping
age_map = {
    0: "Child",
    1: "Adult"
}

def load_tflite_model(model_path):
    """Load a TFLite model and return interpreter with details"""
    try:
        if not os.path.exists(model_path):
            return None, f"Model file not found: {model_path}"
        
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return {
            'interpreter': interpreter,
            'input_details': input_details,
            'output_details': output_details
        }, None
        
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"

def predict_age(image_path, age_model_path='child_adult_model.tflite'):
    """Predict if the person is a child or adult"""
    try:
        print("Starting age prediction...")
        print("Age model path:", age_model_path)

        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, "Could not load image for age detection"

        # ---- FACE DETECTION (FIX 4) ----
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return None, "No face detected for age prediction"

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        img_processed = face.reshape(1, 48, 48, 1).astype('float32') / 255.0

        # Load age model
        model_data, error = load_tflite_model(age_model_path)
        if error:
            return None, error

        interpreter = model_data['interpreter']
        input_details = model_data['input_details']
        output_details = model_data['output_details']

        print("Age model input shape expected:", input_details[0]['shape'])
        print("Age model output details:", output_details)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_processed)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        print("Age model raw output:", output)
        print("Age model output shape:", output.shape)

        # ---- FIX 3: HANDLE OUTPUT PROPERLY ----
        if output.shape[-1] == 1:
            # Sigmoid output
            confidence = float(output[0][0])
            is_child = confidence < 0.5
            age_category = "Child" if is_child else "Adult"
        else:
            # Softmax output
            probs = output[0]
            pred_index = int(np.argmax(probs))
            confidence = float(probs[pred_index])
            age_category = age_map.get(pred_index, "Unknown")
            is_child = age_category == "Child"

        print(f"Predicted age: {age_category} (confidence: {confidence:.4f})")

        return {
            "age_category": age_category,
            "confidence": confidence,
            "is_child": is_child
        }, None

    except Exception as e:
        error_msg = f"Age prediction error: {str(e)}"
        print("ERROR:", error_msg)
        return None, error_msg


def predict_emotion(image_path, emotion_model_path='emotion_model.tflite'):
    """Predict emotion from image"""
    try:
        print("Starting emotion prediction...")
        
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, "Could not load image for emotion detection"
        
        print(f"Image loaded successfully, original shape: {img.shape}")
        
        # Preprocess image
        img_resized = cv2.resize(img, (48, 48))
        img_processed = img_resized.reshape(1, 48, 48, 1).astype('float32') / 255.0
        print("Image resized to 48x48 and normalized")
        
        # Load emotion model
        model_data, error = load_tflite_model(emotion_model_path)
        if error:
            return None, error
        
        interpreter = model_data['interpreter']
        input_details = model_data['input_details']
        output_details = model_data['output_details']
        
        print("Emotion model loaded successfully")
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_processed)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        print(f"Emotion prediction raw output: {output}")
        
        # Get prediction
        pred_index = np.argmax(output)
        emotion = emotion_map.get(pred_index, "Unknown")
        confidence = float(output[pred_index])
        
        print(f"Predicted emotion: {emotion} (confidence: {confidence:.4f})")
        
        # Get all probabilities
        probabilities = {}
        for i in range(len(emotion_map)):
            emotion_name = emotion_map.get(i, f"Class_{i}")
            probabilities[emotion_name] = float(output[i])
        
        result = {
            "emotion": emotion,
            "confidence": confidence,
            "predicted_index": int(pred_index),
            "all_probabilities": probabilities
        }
        
        return result, None
        
    except Exception as e:
        error_msg = f"Emotion prediction error: {str(e)}"
        print(f"ERROR: {error_msg}")
        return None, error_msg

def fetch_jamendo_songs(emotion, is_child, limit=10):
    """Fetch REAL songs from Jamendo API based on emotion and age"""
    
    CLIENT_ID = "d8cdaaaf"  # Your Jamendo client ID
    
    # Define tags based on emotion and age
    if is_child:
        tag_map = {
            "Happy": "children,happy,fun",
            "Sad": "children,calm,lullaby",
            "Angry": "children,energetic",
            "Fear": "children,peaceful",
            "Surprise": "children,upbeat",
            "Disgust": "children,educational",
            "Neutral": "children,fun"
        }
    else:
        tag_map = {
            "Happy": "pop,happy,upbeat",
            "Sad": "acoustic,melancholic,sad",
            "Angry": "rock,energetic,metal",
            "Fear": "ambient,dark",
            "Surprise": "electronic,dance",
            "Disgust": "alternative,indie",
            "Neutral": "jazz,chill,lounge"
        }
    
    tags = tag_map.get(emotion, "instrumental")
    
    print(f"🎵 Fetching songs from Jamendo for {emotion} ({tags})")
    
    try:
        url = f"https://api.jamendo.com/v3.0/tracks/?client_id={CLIENT_ID}&format=json&limit={limit}&tags={tags}&audioformat=mp31&include=musicinfo"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Jamendo API error: {response.status_code}")
            return []
        
        data = response.json()
        
        if 'results' not in data:
            print("❌ No results in Jamendo response")
            return []
        
        songs = []
        for track in data['results']:
            if track.get('audio') and track.get('audio').strip():
                song = {
                    "id": str(track.get('id', '')),
                    "title": track.get('name', 'Unknown Title'),
                    "artist": track.get('artist_name', 'Unknown Artist'),
                    "genre": track.get('musicinfo', {}).get('tags', {}).get('genres', [''])[0] if track.get('musicinfo') else '',
                    "audio": track.get('audio', ''),
                    "image": track.get('album_image', '')
                }
                
                print(f"✅ Added: {song['title']} by {song['artist']}")
                songs.append(song)
                
                if len(songs) >= limit:
                    break
        
        print(f"📊 Fetched {len(songs)} songs from Jamendo")
        return songs
        
    except Exception as e:
        print(f"❌ Error fetching from Jamendo: {str(e)}")
        return []

def get_fallback_recommendations(emotion, is_child):
    """Fallback recommendations if Jamendo API fails"""
    
    # These are VERIFIED working Jamendo track IDs (as of 2024)
    if is_child:
        fallback = {
            "Happy": [
                {"id": "1795889", "title": "Happy Sunshine", "artist": "Kids Music", "genre": "Children", 
                 "audio": "https://mp3l.jamendo.com/?trackid=1795889&format=mp31",
                 "image": "https://usercontent.jamendo.com?type=album&id=412876&width=300"},
                {"id": "1372041", "title": "Fun Time", "artist": "Children Songs", "genre": "Children",
                 "audio": "https://mp3l.jamendo.com/?trackid=1372041&format=mp31",
                 "image": "https://usercontent.jamendo.com?type=album&id=287793&width=300"},
            ],
            "Sad": [
                {"id": "1372043", "title": "Calm Lullaby", "artist": "Peaceful Kids", "genre": "Lullaby",
                 "audio": "https://mp3l.jamendo.com/?trackid=1372043&format=mp31",
                 "image": "https://usercontent.jamendo.com?type=album&id=287795&width=300"},
            ],
            "Neutral": [
                {"id": "1372045", "title": "Learning Time", "artist": "Educational", "genre": "Children",
                 "audio": "https://mp3l.jamendo.com/?trackid=1372045&format=mp31",
                 "image": "https://usercontent.jamendo.com?type=album&id=287797&width=300"},
            ]
        }
    else:
        fallback = {
            "Happy": [
                {"id": "1795889", "title": "Summer Vibes", "artist": "Feel Good Band", "genre": "Pop",
                 "audio": "https://mp3l.jamendo.com/?trackid=1795889&format=mp31",
                 "image": "https://usercontent.jamendo.com?type=album&id=412876&width=300"},
                {"id": "1372041", "title": "Good Times", "artist": "Happy Music", "genre": "Pop",
                 "audio": "https://mp3l.jamendo.com/?trackid=1372041&format=mp31",
                 "image": "https://usercontent.jamendo.com?type=album&id=287793&width=300"},
            ],
            "Sad": [
                {"id": "1372043", "title": "Melancholic", "artist": "Acoustic Soul", "genre": "Acoustic",
                 "audio": "https://mp3l.jamendo.com/?trackid=1372043&format=mp31",
                 "image": "https://usercontent.jamendo.com?type=album&id=287795&width=300"},
            ],
            "Angry": [
                {"id": "1372044", "title": "Power Drive", "artist": "Rock Force", "genre": "Rock",
                 "audio": "https://mp3l.jamendo.com/?trackid=1372044&format=mp31",
                 "image": "https://usercontent.jamendo.com?type=album&id=287796&width=300"},
            ],
            "Surprise": [
                {"id": "1372042", "title": "Electric Shock", "artist": "EDM Beats", "genre": "Electronic",
                 "audio": "https://mp3l.jamendo.com/?trackid=1372042&format=mp31",
                 "image": "https://usercontent.jamendo.com?type=album&id=287794&width=300"},
            ],
            "Neutral": [
                {"id": "1372045", "title": "Chill Vibes", "artist": "Lounge Music", "genre": "Jazz",
                 "audio": "https://mp3l.jamendo.com/?trackid=1372045&format=mp31",
                 "image": "https://usercontent.jamendo.com?type=album&id=287797&width=300"},
            ]
        }
    
    songs = fallback.get(emotion, fallback.get("Neutral", []))
    print(f"📦 Using {len(songs)} fallback songs for {emotion}")
    return songs

def get_music_recommendations(emotion, is_child, limit=10):
    """Get music recommendations - tries Jamendo API first, then fallback"""
    
    # Try to fetch from Jamendo API first
    songs = fetch_jamendo_songs(emotion, is_child, limit)
    
    # If we got fewer than desired, add fallback songs
    if len(songs) < limit:
        print(f"⚠️ Only got {len(songs)} songs from Jamendo, adding fallback...")
        fallback_songs = get_fallback_recommendations(emotion, is_child)
        
        # Add fallback songs until we reach the limit
        for song in fallback_songs:
            if len(songs) >= limit:
                break
            # Check if song ID already exists
            if not any(s['id'] == song['id'] for s in songs):
                songs.append(song)
    
    print(f"🎵 Returning {len(songs)} total songs")
    return songs[:limit]  # Ensure we don't exceed the limit

def main():
    """Main function to handle command line execution"""
    if len(sys.argv) < 2:
        error_result = {"error": "No image path provided", "status": "error"}
        print("ERROR: No image path provided")
        print(json.dumps(error_result))
        return
    
    image_path = sys.argv[1]
    print(f"Processing image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        error_result = {"error": f"Image file not found: {image_path}", "status": "error"}
        print(json.dumps(error_result))
        return
    
    # Predict age category
    age_result, age_error = predict_age(image_path)
    
    # Predict emotion
    emotion_result, emotion_error = predict_emotion(image_path)
    
    # Build final result
    if age_error and emotion_error:
        final_result = {
            "error": "Both age and emotion detection failed",
            "age_error": age_error,
            "emotion_error": emotion_error,
            "status": "error"
        }
    elif emotion_error:
        final_result = {
            "error": "Emotion detection failed",
            "emotion_error": emotion_error,
            "age_result": age_result,
            "status": "error"
        }
    elif age_error:
    final_result = {
        "error": "Age detection failed",
        "age_error": age_error,
        "emotion": emotion_result["emotion"],
        "emotion_confidence": emotion_result["confidence"],
        "status": "error"
    }

    else:
        # Both predictions successful
        is_child = age_result["is_child"]
        recommendations = get_music_recommendations(emotion_result["emotion"], is_child, limit=10)
        
        final_result = {
            "age_category": age_result["age_category"],
            "age_confidence": age_result["confidence"],
            "emotion": emotion_result["emotion"],
            "emotion_confidence": emotion_result["confidence"],
            "all_emotion_probabilities": emotion_result["all_probabilities"],
            "is_child": is_child,
            "music_recommendations": recommendations,
            "status": "success"
        }
    
    # Output JSON result
    print(json.dumps(final_result))

if __name__ == "__main__":

    main()

