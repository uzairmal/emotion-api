import sys
import json
import numpy as np
import cv2
from tensorflow.lite.python.interpreter import Interpreter
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

def predict_age(image_path, age_model_path='age_model.tflite'):
    """Predict if the person is a child or adult"""
    try:
        print("Starting age prediction...")
        
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, "Could not load image for age detection"
        
        # Preprocess image
        img_resized = cv2.resize(img, (48, 48))
        img_processed = img_resized.reshape(1, 48, 48, 1).astype('float32') / 255.0
        
        # Load age model
        model_data, error = load_tflite_model(age_model_path)
        if error:
            return None, error
        
        interpreter = model_data['interpreter']
        input_details = model_data['input_details']
        output_details = model_data['output_details']
        
        print("Age model loaded successfully")
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_processed)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        print(f"Age prediction raw output: {output}")
        
        # Get prediction
        pred_index = np.argmax(output)
        age_category = age_map.get(pred_index, "Unknown")
        confidence = float(output[pred_index])
        
        print(f"Predicted age category: {age_category} (confidence: {confidence:.4f})")
        
        result = {
            "age_category": age_category,
            "confidence": confidence,
            "predicted_index": int(pred_index),
            "is_child": age_category == "Child"
        }
        
        return result, None
        
    except Exception as e:
        error_msg = f"Age prediction error: {str(e)}"
        print(f"ERROR: {error_msg}")
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
        # Age detection failed, but emotion succeeded - use adult songs as default
        print("Warning: Age detection failed, defaulting to adult recommendations")
        is_child = False
        # This would call your music recommendation function if needed
        final_result = {
            "age_category": "Adult (default)",
            "age_detection_error": age_error,
            "emotion": emotion_result["emotion"],
            "emotion_confidence": emotion_result["confidence"],
            "all_emotion_probabilities": emotion_result["all_probabilities"],
            "is_child": is_child,
            "status": "partial_success"
        }
    else:
        # Both predictions successful
        is_child = age_result["is_child"]
        # This would call your music recommendation function if needed
        final_result = {
            "age_category": age_result["age_category"],
            "age_confidence": age_result["confidence"],
            "emotion": emotion_result["emotion"],
            "emotion_confidence": emotion_result["confidence"],
            "all_emotion_probabilities": emotion_result["all_probabilities"],
            "is_child": is_child,
            "status": "success"
        }
    
    # Output JSON result
    print(json.dumps(final_result))

if __name__ == "__main__":
    main()
