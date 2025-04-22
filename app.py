from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from feature_extraction import extract_features
import pickle

app = Flask(__name__)

# Load pre-trained model and data
model = None
word_to_index = None
index_to_word = None
max_length = None
feature_extractor = None

def load_resources():
    global model, word_to_index, index_to_word, max_length
    
    # Load model and vocabulary
    model = load_model('best_model.h5')
    
    with open('vocabulary.pkl', 'rb') as f:
        word_to_index, index_to_word, _ = pickle.load(f)
    
    max_length = 40  # Should match your training max_length

load_resources()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    try:
        # Save the uploaded image temporarily
        temp_path = 'temp_image.jpg'
        image_file.save(temp_path)
        
        # Extract features
        features = extract_features('.', ['temp_image'], save_path=None)
        img_feature = features['temp_image']
        
        # Generate caption
        caption = generate_caption(img_feature)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)})

def generate_caption(image_feature):
    """Generate caption for an image feature."""
    in_text = '<start>'
    caption = []
    
    for _ in range(max_length):
        sequence = [word_to_index.get(word, word_to_index['<unk>']) for word in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        
        yhat = model.predict([np.array([image_feature]), sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        word = index_to_word[yhat]
        
        if word == '<end>':
            break
            
        caption.append(word)
        in_text += ' ' + word
    
    return ' '.join(caption)

if __name__ == '__main__':
    app.run(debug=True)