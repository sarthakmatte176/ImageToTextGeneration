import os
import pickle
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

nltk.download('punkt')

# Load the InceptionV3 model and remove the final classification layer
def get_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model

# Initialize the feature extractor globally
feature_extractor = get_feature_extractor()

# Extract features from an uploaded image
def preprocess_image(img_path):
    try:
        # Load image
        img = image.load_img(img_path, target_size=(299, 299))  # InceptionV3 expects 299x299
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features using the globally initialized model
        features = feature_extractor.predict(x)
        features = np.squeeze(features)  # Shape becomes (2048,)
        return features
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        raise e  # Raising exception so the caller can handle it

def preprocess_captions(captions_file):
    """Preprocess captions from the Flickr8k dataset."""
    with open(captions_file, 'r') as f:
        captions = f.read().strip().split('\n')
    
    # Skip header line if exists
    if 'image,caption' in captions[0]:
        captions = captions[1:]
    
    # Create a dictionary to map image to captions
    image_to_captions = defaultdict(list)
    
    for line in tqdm(captions, desc="Processing captions"):
        parts = line.split(',')
        if len(parts) < 2:
            continue
            
        image_id = parts[0].split('.')[0]  # Remove file extension
        caption = ' '.join(parts[1:]).strip().lower()
        
        # Tokenize and add start/end tokens
        tokens = word_tokenize(caption)
        processed_caption = ['<start>'] + tokens + ['<end>']
        
        image_to_captions[image_id].append(processed_caption)
    
    return image_to_captions

def build_vocabulary(image_to_captions, min_word_count=5):
    """Build vocabulary from captions."""
    word_counts = defaultdict(int)
    
    for captions in image_to_captions.values():
        for caption in captions:
            for word in caption:
                word_counts[word] += 1
    
    # Filter words that appear less than min_word_count
    vocabulary = [word for word, count in word_counts.items() 
                  if count >= min_word_count]
    
    # Add special tokens if not present
    special_tokens = ['<pad>', '<unk>']
    for token in special_tokens:
        if token not in vocabulary:
            vocabulary.append(token)
    
    # Create word to index mapping
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
    index_to_word = {idx: word for idx, word in enumerate(vocabulary)}
    
    return word_to_index, index_to_word, vocabulary

def save_preprocessed_data(data, file_path):
    """Save preprocessed data to file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_preprocessed_data(file_path):
    """Load preprocessed data from file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_image_to_captions(image_to_captions, file_path):
    """Save processed captions to file."""
    with open(file_path, 'wb') as f:
        pickle.dump(image_to_captions, f)

def load_image_to_captions(file_path):
    """Load processed captions from file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
# Example function to post-process the generated caption
def post_process_caption(caption):
    # Remove any <end> tokens
    cleaned_caption = " ".join([word for word in caption.split() if word != "<end>"])
    
    # Capitalize the first letter
    cleaned_caption = cleaned_caption.capitalize()
    
    return cleaned_caption


