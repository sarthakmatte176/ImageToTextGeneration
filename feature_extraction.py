import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tqdm import tqdm

def extract_features(image_dir, image_ids, save_path=None):
    """Extract image features using VGG16."""
    # Load pre-trained VGG16 model without top layer
    model = VGG16(weights='imagenet', include_top=False)
    
    # Dictionary to store features
    features = {}
    
    for img_id in tqdm(image_ids, desc="Extracting features"):
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        
        try:
            # Load and preprocess image
            img = image.load_img(img_path, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            
            # Extract features
            feature = model.predict(img_data)
            
            # Reshape and store
            features[img_id] = np.reshape(feature, (feature.shape[-1],))
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            continue
    
    # Save features if path provided
    if save_path:
        np.save(save_path, features)
    
    return features