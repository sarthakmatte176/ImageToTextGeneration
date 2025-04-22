import os
import pickle
from feature_extraction import extract_features
from preprocessing import preprocess_captions

# Path to the Flickr8k captions file
captions_file = "/Users/sarthakmatte/Image-Captioning-Project/app/templates/Image-Captioning-Project/captions.txt"

# Preprocess the captions
image_to_captions = preprocess_captions(captions_file)

# Make sure the output directory exists
output_dir = "/Users/sarthakmatte/Image-Captioning-Project/data"
os.makedirs(output_dir, exist_ok=True)

# Save the preprocessed captions
with open(os.path.join(output_dir, "image_to_captions.pkl"), "wb") as f:
    pickle.dump(image_to_captions, f)

# Path to the images directory (verify this is correct)
image_dir = "/Users/sarthakmatte/Image-Captioning-Project/app/templates/Image-Captioning-Project/images"

# Get list of image IDs
image_ids = list(image_to_captions.keys())

# Extract features from images
features = extract_features(image_dir, image_ids, save_path="image_features.npy")

print("✅ Feature extraction completed and captions saved.")
