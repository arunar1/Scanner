from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

def load_trained_model(model_path):
    model = Sequential()
    
    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
 
    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
 
    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
 
    model.add(Flatten())
 
    # fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
 
    model.add(Dense(7, activation='softmax'))  # Assuming 7 classes for emotion prediction
    
    # Load weights
    model.load_weights(model_path)
    
    return model

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    processed_img = np.asarray(resized_img, dtype=np.float32) / 255.0
    processed_img = processed_img.reshape(1, 48, 48, 1)  # Reshape for model input
    return processed_img

def predict_emotion(image_path, model):
    processed_img = preprocess_image(image_path)
    predicted_probabilities = model.predict(processed_img)
    predicted_class = np.argmax(predicted_probabilities)
    return predicted_class

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": ""}})

classes = ['camera', 'Keyboards', 'smartwatch', 'Mobile', 'Mouses', 'laptop', 'TV' ]  # Define the classes for prediction

@app.route("/")
def index():
    return "Welcome to the E-waste Detection"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({'error': 'No selected file'}), 400

    # Save the image temporarily
    image_path = "temp_image.jpg"
    file.save(image_path)

    # Get the predicted class
    predicted_class_index = main(image_path)
    predicted_class_label = classes[predicted_class_index]

    # Remove the temporary image file
    os.remove(image_path)

    return jsonify({"Prediction": predicted_class_label,"class":predicted_class_index})

def main(image_path):
    # Path to the trained model
    model_path = "model1.h5"
    # Load the trained model
    model = load_trained_model(model_path)
 
    # Get the predicted class
    predicted_class = predict_emotion(image_path, model)
    return predicted_class

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
