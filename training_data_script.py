import numpy as np
import pickle

from data_preparation import create_training_sequences  # Ensure this function exists and is imported correctly

# Load image features
print("🔍 Loading image features...")
features = np.load("image_features.npy", allow_pickle=True).item()

# Load preprocessed captions
print("📝 Loading preprocessed captions...")
with open("/Users/sarthakmatte/Image-Captioning-Project/data/image_to_captions.pkl", "rb") as f:
    image_to_captions = pickle.load(f)

# Load tokenizer data
print("🔤 Loading word index...")
with open("/Users/sarthakmatte/Image-Captioning-Project/data/word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)

# Load max length of captions
print("📏 Loading max caption length...")
with open("/Users/sarthakmatte/Image-Captioning-Project/data/max_length.pkl", "rb") as f:
    max_length = pickle.load(f)

# Calculate vocab_size
vocab_size = len(word_to_index) + 1  # Add 1 for the padding token

# Create training sequences
print("⚙️ Creating training sequences...")
X1, X2, y = create_training_sequences(image_to_captions, word_to_index, features, max_length, vocab_size)

# Save training data (optional)
np.save("X1.npy", X1)
np.save("X2.npy", X2)
np.save("y.npy", y)

print("✅ Training data prepared and saved successfully!")
