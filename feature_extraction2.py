import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Load the InceptionV3 model pre-trained on ImageNet
model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # Resize image to fit InceptionV3 input size
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for InceptionV3
    return model.predict(img_array)  # Get the features

# Directory containing your images
img_dir = '/Users/sarthakmatte/Image-Captioning-Project/app/templates/Image-Captioning-Project/images'

# Dictionary to store image features
features_dict = {}

# Loop through each image in the directory, extract features, and save them
for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    features_dict[img_name] = extract_features(img_path)

# Save the features to a .npy file
np.save('image_features.npy', features_dict)

print(f"Image features saved to 'image_features.npy'.")
