from preprocessing import preprocess_captions, build_vocabulary, save_preprocessed_data
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

# Path to the captions file
captions_file = '/Users/sarthakmatte/Image-Captioning-Project/app/templates/Image-Captioning-Project/captions.txt'

# Preprocess captions
image_to_captions = preprocess_captions(captions_file)

# Build vocabulary
word_to_index, index_to_word, vocabulary = build_vocabulary(image_to_captions)

# Save vocabulary
save_preprocessed_data((word_to_index, index_to_word, vocabulary), 'vocabulary.pkl')