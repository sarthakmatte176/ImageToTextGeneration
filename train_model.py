import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_model
from preprocessing import load_preprocessed_data
from data_preparation import prepare_training_data
import pickle

# Try to import load_image_to_captions from preprocessing
try:
    from preprocessing import load_image_to_captions
except ImportError:
    print("load_image_to_captions not found, falling back to preprocess_captions...")
    from preprocessing import preprocess_captions as load_image_to_captions

# Load image features
features = np.load('/Users/sarthakmatte/Image-Captioning-Project/app/templates/image_features.npy', allow_pickle=True).item()
print("Loaded image features:", type(features), "with", len(features), "entries.")

if not isinstance(features, dict) or len(features) == 0:
    raise ValueError("Image features are not loaded correctly. Please check the .npy file.")

# Load vocabulary
word_to_index, index_to_word, _ = load_preprocessed_data('vocabulary.pkl')

# Try loading preprocessed captions, fallback to processing
try:
    image_to_captions = load_image_to_captions('image_to_captions.pkl')
except (FileNotFoundError, TypeError):
    print("Preprocessed captions not found, processing from scratch...")
    from preprocessing import preprocess_captions
    image_to_captions = preprocess_captions(
        '/Users/sarthakmatte/Image-Captioning-Project/app/templates/Image-Captioning-Project/captions.txt')
    with open('image_to_captions.pkl', 'wb') as f:
        pickle.dump(image_to_captions, f)

# Prepare training data
from data_preparation import prepare_training_data
X_images, X_seqs, y_words = prepare_training_data(
    image_to_captions,
    word_to_index,
    features,
    max_length=40
)

# Squeeze to remove extra dimension (if present)
X_images = np.squeeze(X_images)
print("Shape of X_images after squeezing:", X_images.shape)

# Ensure that X_images is not empty
if X_images.shape[0] == 0:
    raise ValueError("X_images is empty. Check the image features or the captions.")
# Create and train model
model = create_model(
    vocab_size=len(word_to_index) +1,
    max_length=40,
    feature_size=X_images.shape[1]
)

history = model.fit(
    [X_images, X_seqs],
    y_words,
    epochs=1,
    batch_size=8,
    validation_split=0.2,
    callbacks=[
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=3)
    ]
)
