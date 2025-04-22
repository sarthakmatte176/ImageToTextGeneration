import os
import pickle
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

nltk.download('punkt')

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
    
import pickle
from collections import defaultdict

# Add these functions to your existing preprocessing.py
def save_image_to_captions(image_to_captions, file_path):
    """Save processed captions to file."""
    with open(file_path, 'wb') as f:
        pickle.dump(image_to_captions, f)

def load_image_to_captions(file_path):
    """Load processed captions from file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)