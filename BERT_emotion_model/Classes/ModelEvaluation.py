### This class for evaluate the model ###
# Import needed libs
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluation:
    
    def __init__(self, model):

        self.model = model
    
    def model_performance(self, history, epochs):

        # Extract needed variables
        tr_acc = history.history['accuracy']
        tr_loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        
        epochs_range = [i + 1 for i in range(len(tr_acc))]

        # Plot training history
        plt.figure(figsize=(20, 8))
        plt.style.use('fivethirtyeight')
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, tr_loss, 'r', label='Training Loss')
        plt.plot(epochs_range, val_loss, 'g', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, tr_acc, 'r', label='Training Accuracy')
        plt.plot(epochs_range, val_acc, 'g', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def evaluate_model_with_confusion_matrix(self, test_dataset, label_names):

        # Get the true labels from the test set
        true_labels = []
        for batch in test_dataset:
            _, labels = batch
            true_labels.extend(labels.numpy())  # Add true labels to the list

        true_labels = np.array(true_labels)

        # Predict labels using the fine-tuned model
        pred_labels = []
        for batch in test_dataset:
            inputs, _ = batch
            predictions = self.model.predict(inputs)
            predicted_classes = np.argmax(predictions, axis=1)
            pred_labels.extend(predicted_classes)

        pred_labels = np.array(pred_labels)

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(true_labels, pred_labels)

        # Plot the confusion matrix with the specified labels
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Print classification report with label names
        print(classification_report(true_labels, pred_labels, target_names=label_names))



