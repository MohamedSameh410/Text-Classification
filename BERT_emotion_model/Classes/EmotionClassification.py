### This class for classify the emotion in the text ###
#Import needed libs
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer


class EmotionClassification:
    def __init__(self, dir_path):

        self.dir_path = dir_path
        self.model = TFAutoModelForSequenceClassification.from_pretrained(dir_path)
        self.tokenizer = AutoTokenizer.from_pretrained(dir_path)
    
    
    def predict_emotion(self, text):
        
        inputs = self.tokenizer(text, return_tensors='tf', truncation=True, padding=True)
        outputs = self.model(inputs)
        logits = outputs.logits  # Access logits directly
        predicted_class = tf.argmax(logits, axis=1).numpy()[0]
        emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

        return emotion_labels[predicted_class]

