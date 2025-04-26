import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def prepare_training_data(image_to_captions, word_to_index, features, max_length):
    X1, X2, y = list(), list(), list()

    # Fix potential key mismatch (remove .jpg from keys if needed)
    fixed_features = {key.split('.')[0]: value for key, value in features.items()}

    matched = 0
    missing = 0

    vocab_size = len(word_to_index) + 1  # +1 for padding token 0

    for image_id, captions in image_to_captions.items():
        feature = fixed_features.get(image_id)
        if feature is None:
            missing += 1
            continue

        matched += 1

        for caption in captions:
            # Handle list vs. string for captions
            words = caption if isinstance(caption, list) else caption.split()
            seq = [word_to_index.get(word, None) for word in words]  # Handle missing words explicitly

            # Skip captions with unknown words
            seq = [w for w in seq if w is not None]

            for i in range(1, len(seq)):
                in_seq, out_word = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_word = to_categorical([out_word], num_classes=vocab_size)[0]

                X1.append(feature)
                X2.append(in_seq)
                y.append(out_word)

    print(f"[INFO] Matched images: {matched}")
    print(f"[INFO] Skipped images with missing features: {missing}")
    print(f"[INFO] Final training pairs: {len(X1)}")

    return np.array(X1), np.array(X2), np.array(y)

def create_training_sequences(image_to_captions, word_to_index, features, max_length, vocab_size):
    X1, X2, y = [], [], []

    for image_id, captions in image_to_captions.items():
        feature = features.get(image_id)
        if feature is None:
            continue

        for caption in captions:
            words = caption if isinstance(caption, list) else caption.split()
            seq = [word_to_index.get(word, None) for word in words]

            # Skip captions with unknown words
            seq = [w for w in seq if w is not None]

            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)
