import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_model
from preprocessing import load_preprocessed_data
from data_preparation import prepare_training_data
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
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
vocab_size = len(word_to_index) + 1
max_length = 40

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

# Define the data generator
def data_generator(image_to_captions, features, word_to_index, max_length, vocab_size, batch_size):
    while True:
        X1, X2, y = list(), list(), list()
        n = 0
        for image_id, captions in image_to_captions.items():
            feature = features.get(image_id)
            if feature is None:
                continue
            for caption in captions:
                seq = [word_to_index[word] for word in caption.split() if word in word_to_index]
                for i in range(1, len(seq)):
                    in_seq, out_word = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_word = to_categorical([out_word], num_classes=vocab_size)[0]

                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_word)
                    n += 1
                    if n == batch_size:
                        yield [np.array(X1), np.array(X2)], np.array(y)
                        X1, X2, y = list(), list(), list()
                        n = 0

# Build model
model = create_model(
    vocab_size=vocab_size,
    max_length=max_length,
    feature_size=2048  # Your image feature vector size
)

# Define training parameters
batch_size = 32
total_samples = sum([len(caps) for caps in image_to_captions.values()])
steps = total_samples // batch_size

# Train the model using the generator
history = model.fit(
    data_generator(image_to_captions, features, word_to_index, max_length, vocab_size, batch_size),
    steps_per_epoch=steps,
    epochs=5,
    callbacks=[
        ModelCheckpoint('best_model.h5', monitor='loss', save_best_only=True),
        EarlyStopping(monitor='loss', patience=2)
    ]
)
