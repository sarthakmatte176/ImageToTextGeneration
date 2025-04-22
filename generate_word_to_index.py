import pickle
from collections import Counter
from preprocessing import preprocess_captions
import os
# Load captions
captions_file = "/Users/sarthakmatte/Image-Captioning-Project/app/templates/Image-Captioning-Project/captions.txt"
image_to_captions = preprocess_captions(captions_file)

# Build vocabulary
def build_vocabulary(image_to_captions, min_word_freq=5):
    word_counts = Counter()

    for captions in image_to_captions.values():
        for caption in captions:
            for word in caption:
                word_counts[word] += 1

    vocab = [word for word, count in word_counts.items() if count >= min_word_freq]

    word_to_index = {word: idx + 1 for idx, word in enumerate(vocab)}
    word_to_index['<pad>'] = 0
    word_to_index['<start>'] = len(word_to_index)
    word_to_index['<end>'] = len(word_to_index)
    word_to_index['<unk>'] = len(word_to_index)

    return word_to_index

# Build and save
word_to_index = build_vocabulary(image_to_captions)

# Make sure directory exists
output_path = "/Users/sarthakmatte/Image-Captioning-Project/data"
os.makedirs(output_path, exist_ok=True)

# Save the dictionary
with open(os.path.join(output_path, "word_to_index.pkl"), "wb") as f:
    pickle.dump(word_to_index, f)

print("✅ word_to_index.pkl generated successfully.")

# Path to your captions file
captions_file = "/Users/sarthakmatte/Image-Captioning-Project/app/templates/Image-Captioning-Project/captions.txt"

# Preprocess captions
image_to_captions = preprocess_captions(captions_file)

# Calculate max caption length
max_length = max(len(caption) for captions in image_to_captions.values() for caption in captions)

# Save max_length to pkl
with open("/Users/sarthakmatte/Image-Captioning-Project/data/max_length.pkl", "wb") as f:
    pickle.dump(max_length, f)

print(f"✅ Max caption length saved: {max_length}")
