# Text Classification
## Overview
This repository contains a model fine-tuned on a pre-trained BERT (Bidirectional Encoder Representations from Transformers) architecture for the task of emotion classification. The model is designed to categorize text into predefined emotion labels, making it useful for applications such as sentiment analysis, chatbots, customer feedback systems, and more.

## Features
- **Pre-trained BERT Model:** Built on top of the BERT architecture, leveraging its ability to understand context in natural language.
- **Emotion Classification:** Fine-tuned to classify text into multiple emotion categories (e.g., happiness, sadness, anger, etc.).
- **Custom Dataset:** The model was trained on a custom dataset that includes labeled emotional content from various sources.
- **Text Input:** Accepts raw text input and outputs the predicted emotion category.

## Getting Started

### Prerequisites
All prerequisites in `requirements.txt` file

### Installation
Follow the steps below to set up the project on your local machine:
1. Clone the repository:
   
   ```bash
   git clone https://github.com/MohamedSameh410/Text-Classification.git
   cd Text-Classification
   ```
2. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```
3. Run the command-line tool to use the model

   - **If you want to use the model through the UI:**

     ```bash
     cd BERT_emotion_model
     python app.py
     ```
   - **If you want to use the model through the cmd:**

     ```bash
     cd BERT_emotion_model
     python test_model.py
     ```

## Emotion Labels
The model can classify text into the following emotion categories:

- Joy
- Sadness
- Anger
- Surprise
- Fear
- Love

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.
