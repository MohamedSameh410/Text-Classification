from flask import Flask, request, jsonify, render_template
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np


dir_path = 'emotion_model'
loaded_model = TFAutoModelForSequenceClassification.from_pretrained(dir_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(dir_path)

def predict_emotion(text):
    inputs = loaded_tokenizer(text, return_tensors='tf', truncation=True, padding=True)
    outputs = loaded_model(inputs)
    logits = outputs.logits  # Access logits directly
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    return emotion_labels[predicted_class]

# Initialize the Flask app
app = Flask(__name__)

# Define the main route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form  # Get the form data
    text = data.get("text")  # Extract the text from the input

    if not text:
        return jsonify({"error": "Please provide text input!"}), 400  # Return an error response

    try:
        print(f"Input Text: {text}")  # Debugging statement
        # Make prediction using the emotion classifier
        predicted_emotion = predict_emotion(text)

        # Return the predicted emotion
        return jsonify({"prediction": predicted_emotion})

    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging statement
        return jsonify({"error": str(e)}), 500  # Return an error response if something goes wrong
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
