# Import needed libs
from Classes.EmotionClassification import EmotionClassification

# Get instance from the class and load the model
emotion_classifier = EmotionClassification('emotion_model')

# Get input from user
text = input("Enter your text here: ")

# Get the predicted emotion
predicted_emotion = emotion_classifier.predict_emotion(text)
print(f"Predicted Emotion: {predicted_emotion}")