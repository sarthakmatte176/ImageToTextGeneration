from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Add, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def create_model(vocab_size, max_length, feature_size=4096, embedding_dim=256, lstm_units=256):
    """Create image captioning model."""
    # Image feature input
    inputs1 = Input(shape=(feature_size,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(embedding_dim, activation='relu')(fe1)
    
    # Sequence input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(lstm_units)(se2)
    
    # Decoder
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(lstm_units, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Create model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    
    return model

def train_model(model, X_images, X_seqs, y_words, epochs=20, batch_size=64):
    """Train the captioning model."""
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        [X_images, X_seqs],
        y_words,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[checkpoint, early_stopping]
    )
    
    return history