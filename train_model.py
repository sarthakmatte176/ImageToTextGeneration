import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_model
from preprocessing import load_preprocessed_data
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
import os
from keras import backend as K

# Attempt to import caption loader
try:
    from preprocessing import load_image_to_captions
except ImportError:
    print("load_image_to_captions not found, using preprocess_captions instead...")
    from preprocessing import preprocess_captions as load_image_to_captions

# Load pre-extracted image features
features = np.load('/Users/sarthakmatte/Image-Captioning-Project/app/templates/image_features.npy', allow_pickle=True).item()
print("Loaded image features:", type(features), "with", len(features), "entries.")

# Debug feature shapes
for k in list(features.keys())[:5]:
    print(f"Feature shape for {k}: {features[k].shape}")

if not isinstance(features, dict) or len(features) == 0:
    raise ValueError("Image features are not loaded correctly. Please check the .npy file path or content.")

# Load vocabulary
word_to_index, index_to_word, _ = load_preprocessed_data('/Users/sarthakmatte/Image-Captioning-Project/app/templates/vocabulary.pkl')
vocab_size = len(word_to_index) + 1  # Adding 1 for padding token
max_length = 40  # Max length of the caption sequences

# Load or preprocess captions
try:
    image_to_captions = load_image_to_captions('image_to_captions.pkl')
except (FileNotFoundError, TypeError):
    print("Preprocessed captions not found, creating from scratch...")
    from preprocessing import preprocess_captions
    image_to_captions = preprocess_captions('/Users/sarthakmatte/Image-Captioning-Project/app/templates/Image-Captioning-Project/captions.txt')
    with open('image_to_captions.pkl', 'wb') as f:
        pickle.dump(image_to_captions, f)

# Limit to first 400 image-caption pairs for faster testing
image_to_captions = dict(list(image_to_captions.items())[:400])

# Strip file extensions from keys in features dict
features = {k.split('.')[0]: v for k, v in features.items()}

# Filter only matching keys (only consider features that have corresponding captions)
features = {k: features[k] for k in image_to_captions.keys() if k in features}

print("Matching keys:", len(features))
print("Example key from image_to_captions:", next(iter(image_to_captions.keys())))
print("Example keys from features:", list(features.keys())[:5])

def data_generator(image_to_captions, features, word_to_index, max_length, vocab_size, batch_size):
    while True:
        X1, X2, y = [], [], []
        n = 0
        for image_id, captions in image_to_captions.items():
            feature = features.get(image_id)
            if feature is None:
                continue
            # Ensure feature has shape (2048,)
            if feature.ndim > 1:
                feature = np.squeeze(feature)  # Remove extra dimensions
            if feature.shape != (2048,):
                raise ValueError(f"Feature for {image_id} has unexpected shape {feature.shape}")
            for caption in captions:
                seq = [word_to_index[word] for word in caption if word in word_to_index]
                for i in range(1, len(seq)):
                    in_seq, out_word = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_word = to_categorical([out_word], num_classes=vocab_size)[0]
                    X1.append(feature)  # Add feature (2048,)
                    X2.append(in_seq)   # Add in_seq (padded sequence)
                    y.append(out_word)  # One-hot encoded output word
                    n += 1
                    if n == batch_size:
                        # Stack into batches
                        X1_batch = np.stack(X1, axis=0)  # Shape: (batch_size, 2048)
                        X2_batch = np.array(X2)  # Shape: (batch_size, max_length)
                        y_batch = np.array(y)  # Shape: (batch_size, vocab_size)
                        yield ((X1_batch, X2_batch), y_batch)  # Yield inputs as tuple
                        X1, X2, y = [], [], []
                        n = 0

# Debug generator output shapes
for inputs, outputs in data_generator(image_to_captions, features, word_to_index, max_length, vocab_size, batch_size=8):
    X1_batch, X2_batch = inputs  # Unpack tuple
    print("Generator output shapes:")
    print("X1_batch (image features):", X1_batch.shape)  # Should be (8, 2048)
    print("X2_batch (input sequences):", X2_batch.shape)  # Should be (8, 40)
    print("y_batch (output words):", outputs.shape)  # Should be (8, vocab_size)
    print("Input structure:", type(inputs), "Output structure:", type(outputs))
    break

# Create model
model = create_model(vocab_size=vocab_size, max_length=max_length, feature_size=2048)

# Training configuration
batch_size = 8
total_samples = sum(len(caps) for caps in image_to_captions.values())
steps = max(1, total_samples // batch_size)

# Debugging output
print("Total samples:", total_samples)
print("Steps per epoch:", steps)
print("Filtered features:", len(features))
print("Total captions:", len(image_to_captions))

# Create TensorFlow Dataset from generator
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(image_to_captions, features, word_to_index, max_length, vocab_size, batch_size),
    output_signature=(
        (
            tf.TensorSpec(shape=(None, 2048), dtype=tf.float32),  # Batched image features
            tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)  # Batched input sequences
        ),
        tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)  # Batched output words
    )
)

# Define callbacks for saving the best model
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train model
print("Starting training...")
try:
    history = model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=steps,
        callbacks=[checkpoint],
        verbose=1
    )
    print("Training completed successfully")
except Exception as e:
    print(f"Training failed: {e}")
    raise

# Save the final model
model.save('final_model.h5')
print("Final model saved as 'final_model.h5'")
if os.path.exists('final_model.h5'):
    print("Confirmed: 'final_model.h5' exists, size:", os.path.getsize('final_model.h5'), "bytes")
else:
    print("Error: 'final_model.h5' was not saved")

# Cleanup session
K.clear_session()

# Optional: Enable memory growth for GPU (only needed if GPU is used)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print("GPU memory setup failed:", e)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_model
from preprocessing import load_preprocessed_data
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
from keras import backend as K

# Attempt to import caption loader
try:
    from preprocessing import load_image_to_captions
except ImportError:
    print("load_image_to_captions not found, using preprocess_captions instead...")
    from preprocessing import preprocess_captions as load_image_to_captions

# Load pre-extracted image features
features = np.load('/Users/sarthakmatte/Image-Captioning-Project/app/templates/image_features.npy', allow_pickle=True).item()
print("Loaded image features:", type(features), "with", len(features), "entries.")

# Debug feature shapes
for k in list(features.keys())[:5]:
    print(f"Feature shape for {k}: {features[k].shape}")

if not isinstance(features, dict) or len(features) == 0:
    raise ValueError("Image features are not loaded correctly. Please check the .npy file path or content.")

# Load vocabulary
word_to_index, index_to_word, _ = load_preprocessed_data('/Users/sarthakmatte/Image-Captioning-Project/app/templates/vocabulary.pkl')
vocab_size = len(word_to_index) + 1  # Adding 1 for padding token
max_length = 40  # Max length of the caption sequences

# Load or preprocess captions
try:
    image_to_captions = load_image_to_captions('image_to_captions.pkl')
except (FileNotFoundError, TypeError):
    print("Preprocessed captions not found, creating from scratch...")
    from preprocessing import preprocess_captions
    image_to_captions = preprocess_captions('/Users/sarthakmatte/Image-Captioning-Project/app/templates/Image-Captioning-Project/captions.txt')
    with open('image_to_captions.pkl', 'wb') as f:
        pickle.dump(image_to_captions, f)

# Limit to first 400 image-caption pairs for faster testing
image_to_captions = dict(list(image_to_captions.items())[:400])

# Strip file extensions from keys in features dict
features = {k.split('.')[0]: v for k, v in features.items()}

# Filter only matching keys (only consider features that have corresponding captions)
features = {k: features[k] for k in image_to_captions.keys() if k in features}

print("Matching keys:", len(features))
print("Example key from image_to_captions:", next(iter(image_to_captions.keys())))
print("Example keys from features:", list(features.keys())[:5])

def data_generator(image_to_captions, features, word_to_index, max_length, vocab_size, batch_size):
    while True:
        X1, X2, y = [], [], []
        n = 0
        for image_id, captions in image_to_captions.items():
            feature = features.get(image_id)
            if feature is None:
                continue
            # Ensure feature has shape (2048,)
            if feature.ndim > 1:
                feature = np.squeeze(feature)  # Remove extra dimensions
            if feature.shape != (2048,):
                raise ValueError(f"Feature for {image_id} has unexpected shape {feature.shape}")
            for caption in captions:
                seq = [word_to_index[word] for word in caption if word in word_to_index]
                for i in range(1, len(seq)):
                    in_seq, out_word = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_word = to_categorical([out_word], num_classes=vocab_size)[0]
                    X1.append(feature)  # Add feature (2048,)
                    X2.append(in_seq)   # Add in_seq (padded sequence)
                    y.append(out_word)  # One-hot encoded output word
                    n += 1
                    if n == batch_size:
                        # Stack into batches
                        X1_batch = np.stack(X1, axis=0)  # Shape: (batch_size, 2048)
                        X2_batch = np.array(X2)  # Shape: (batch_size, max_length)
                        y_batch = np.array(y)  # Shape: (batch_size, vocab_size)
                        yield ((X1_batch, X2_batch), y_batch)  # Yield inputs as tuple
                        X1, X2, y = [], [], []
                        n = 0

# Debug generator output shapes
for inputs, outputs in data_generator(image_to_captions, features, word_to_index, max_length, vocab_size, batch_size=8):
    X1_batch, X2_batch = inputs  # Unpack tuple
    print("Generator output shapes:")
    print("X1_batch (image features):", X1_batch.shape)  # Should be (8, 2048)
    print("X2_batch (input sequences):", X2_batch.shape)  # Should be (8, 40)
    print("y_batch (output words):", outputs.shape)  # Should be (8, vocab_size)
    print("Input structure:", type(inputs), "Output structure:", type(outputs))
    break

# Create model
model = create_model(vocab_size=vocab_size, max_length=max_length, feature_size=2048)

# Training configuration
batch_size = 8
total_samples = sum(len(caps) for caps in image_to_captions.values())
steps = max(1, total_samples // batch_size)

# Debugging output
print("Total samples:", total_samples)
print("Steps per epoch:", steps)
print("Filtered features:", len(features))
print("Total captions:", len(image_to_captions))

# Create TensorFlow Dataset from generator
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(image_to_captions, features, word_to_index, max_length, vocab_size, batch_size),
    output_signature=(
        (
            tf.TensorSpec(shape=(None, 2048), dtype=tf.float32),  # Batched image features
            tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)  # Batched input sequences
        ),
        tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)  # Batched output words
    )
)

# Define callbacks for saving the best model
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train model
history = model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=steps,
    callbacks=[checkpoint],
    verbose=1
)

# Save the final model
model.save('final_model.h5')
print("Final model saved as 'final_model.h5'")

# Cleanup session
K.clear_session()

# Optional: Enable memory growth for GPU (only needed if GPU is used)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print("GPU memory setup failed:", e)"""
