import pickle

# Load word_to_index
print("🔤 Loading word index...")
with open("/Users/sarthakmatte/Image-Captioning-Project/data/word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)

# Now, you can access word_to_index and calculate vocab_size
vocab_size = len(word_to_index) + 1  # Add 1 for the padding token

# You can now proceed with the rest of your training logic
