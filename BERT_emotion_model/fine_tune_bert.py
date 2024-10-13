# Import needed libs
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset
from Classes.BERTForClassification import BERTForClassification
from Classes.ModelEvaluation import ModelEvaluation

# Load the dataset
emotions = load_dataset('SetFit/emotion')

# Load the BERT model
model = TFAutoModel.from_pretrained("bert-base-uncased")
model = model.to('cuda')

# Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# setting 'input_ids', 'attention_mask', 'token_type_ids', and 'label' to the tensorflow format
emotions_encoded.set_format('tf', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])


# Group the inputs into a single dictionary
def order(inp):

    data = list(inp.values())
    return {
        'input_ids': data[1],
        'attention_mask': data[2],
        'token_type_ids': data[3]
    }, data[0]



# setting the batch size
BATCH_SIZE = 64

# Converting train split of `emotions_encoded` to tensorflow format
train_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['train'][:])
# Set batch_size and shuffle
train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
# map the `order` function
train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

# Doing the same for val set
val_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['test'][:])
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

# Doing the same for test set
test_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['test'][:])
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)


#... Fine-tune the model ...
# Compile the BERT model
classifier = BERTForClassification(model, num_classes=6)

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the fine-tuned BERT
epochs = 5

history = classifier.fit(
    train_dataset,
    epochs = epochs,
    validation_data = val_dataset
)

# Git instance from ModelEvaluation class
model_eval = ModelEvaluation(classifier)

# Visualize the model performance
model_eval.model_performance(history, epochs=5)

# Evaluate the model
classifier.evaluate(test_dataset)

# Set the label names
label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Display the confusion matrix & print the classification report
model_eval.evaluate_model_with_confusion_matrix(test_dataset, label_names)

# Save the model to the working directory
dir_path = 'bert_emotion_classifier'
classifier.save_the_model(dir_path)


