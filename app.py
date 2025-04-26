import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import create_model
from preprocessing import preprocess_image, load_preprocessed_data
import logging
from flask_cors import CORS  # Correct import for CORS

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)  # Enable CORS by calling CORS(app)

# Global variables
model = None
features = None
word_to_index = None
index_to_word = None
max_length = 40

def load_resources():
    global model, features, word_to_index, index_to_word

    # Load pre-extracted image features
    features_path = 'app/templates/image_features.npy'
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    features = np.load(features_path, allow_pickle=True).item()

    # Strip file extensions from feature keys if needed
    features = {k.split('.')[0]: v for k, v in features.items()}

    # Load vocabulary
    vocab_path = 'app/templates/vocabulary.pkl'
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    word_to_index, index_to_word, _ = load_preprocessed_data(vocab_path)

    # Create model
    vocab_size = len(word_to_index) + 1  # Adding 1 for padding
    feature_size = 2048
    model = create_model(vocab_size=vocab_size, max_length=max_length, feature_size=feature_size)

    # Load model weights (not entire model because of compatibility issues)
    weights_path = 'final_model.h5'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model file not found: {weights_path}")
    model.load_weights(weights_path)

    logging.info("Resources loaded successfully!")

def generate_caption(image_feature):
    # Start the caption with start token
    in_text = ['startseq']

    try:
        for i in range(max_length):
            sequence = [word_to_index.get(word, 0) for word in in_text]
            sequence = pad_sequences([sequence], maxlen=max_length)

            # Predict the next word
            yhat = model.predict([np.expand_dims(image_feature, axis=0), sequence], verbose=0)
            yhat = np.argmax(yhat)

            word = index_to_word.get(yhat, None)
            if word is None:
                break
            in_text.append(word)

            if word == 'endseq':
                break
    except Exception as e:
        logging.error(f"Error generating caption: {e}")
        return "Error generating caption."
    
    caption = ' '.join(in_text[1:-1])  # Remove 'startseq' and 'endseq'
    return caption

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    try:
        if request.method == 'POST':
            file = request.files['image']
            if file:
                filepath = os.path.join('uploads', file.filename)
                os.makedirs('uploads', exist_ok=True)
                file.save(filepath)
                logging.info(f"Image uploaded to: {filepath}")

                # Preprocess the image (Add try-except here)
                try:
                    image_feature = preprocess_image(filepath)
                    logging.info(f"Image feature shape: {image_feature.shape}")
                except Exception as e:
                    logging.error(f"Error preprocessing image: {e}")
                    return render_template('index.html', caption="Error processing image.")

                # Generate caption
                caption = generate_caption(image_feature)
                logging.info(f"Generated caption: {caption}")
            else:
                logging.warning("No image uploaded.")
        else:
            logging.info("GET request received.")
    except Exception as e:
        logging.error(f"Error during caption generation: {e}")
        caption = f"Error: {str(e)}"

    return render_template('index.html', caption=caption)


if __name__ == '__main__':
    try:
        load_resources()
        app.run(debug=True, port=5001)  # Changed port to 5001
    except Exception as e:
        logging.error(f"Error starting the server: {e}")
