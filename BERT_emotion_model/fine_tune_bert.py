# Import needed libs
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from Classes.ModelEvaluation import ModelEvaluation

# Load the dataset
emotions = load_dataset('SetFit/emotion')

# Load the BERT model
model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model = model.to('cuda')

# Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True)

# Define the data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")


# Setting the batch size
batch_size = 32

# Convert dataset to tf.data.Dataset
def convert_to_tf_dataset(split, batch_size):
    return emotions_encoded[split].to_tf_dataset(
        columns=['input_ids', 'attention_mask', 'token_type_ids'],
        label_cols=['label'],
        shuffle=(split == 'train'),
        batch_size=batch_size,
        collate_fn=data_collator
    )

# Convert the train split
train_dataset = convert_to_tf_dataset('train')

# Convert the val split
val_dataset = convert_to_tf_dataset('validation')

# Convert the test split
test_dataset = convert_to_tf_dataset('test')

#... Fine-tune the model ...
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
epochs = 3

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Git instance from ModelEvaluation class
model_eval = ModelEvaluation(model)

# Visualize the model performance
model_eval.model_performance(history, epochs = epochs)

# Evaluate the model
model.evaluate(test_dataset)

# Set the label names
label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Display the confusion matrix & print the classification report
model_eval.evaluate_model_with_confusion_matrix(test_dataset, label_names)

# Save the model and tokenizer
model.save_pretrained('emotion_model')
tokenizer.save_pretrained('emotion_model')


