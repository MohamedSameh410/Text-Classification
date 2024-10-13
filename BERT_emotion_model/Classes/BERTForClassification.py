### This calss for fine-tune BERT and add an decoder layer (softmax) ###
# Import the needed libs
import tensorflow as tf

class BERTForClassification(tf.keras.Model):
    
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        # Extract input_ids, attention_mask, and token_type_ids from the inputs
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        
        # Pass them to the BERT model
        bert_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        return self.fc(bert_output)

    def save_the_model(self, dir_path):
        # Save the BERT model
        self.bert.save_pretrained(dir_path)
