

import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical

# ------------------- STEP 1: Load and Clean Data 
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Make everything small letters and remove special characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)   # keep only letters and spaces
    
    print("✅ Text loaded successfully!")
    print(f"Total characters: {len(text)}")
    
    return text


# -------------------------------------- STEP 2: Prepare Data for Model
def prepare_data(text, seq_length=50):
    # Create vocabulary (all unique characters)
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Vocabulary size: {len(chars)} characters")
    
    # Create training sequences
    X = []
    y = []
    
    for i in range(0, len(text) - seq_length):
        sequence = text[i:i + seq_length]
        next_char = text[i + seq_length]
        
        X.append([char_to_int[char] for char in sequence])
        y.append(char_to_int[next_char])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = to_categorical(y, num_classes=len(chars))   # one-hot encoding
    
    # Reshape for LSTM (samples, time_steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, char_to_int, int_to_char, len(chars)


# ------------------------------------------- STEP 3: Build the LSTM Model
def create_model(vocab_size, seq_length):
    model = Sequential()
    
    model.add(Embedding(vocab_size, 100, input_length=seq_length))
    model.add(LSTM(128))                    # Main memory layer
    model.add(Dense(64, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    print("✅ Model created!")
    model.summary()
    return model


# ----------------------------------------------- STEP 4: Generate New Text 
def generate_text(model, seed_text, char_to_int, int_to_char, n_chars=300):
    # Clean seed text
    seed_text = re.sub(r'[^a-z\s]', '', seed_text.lower())
    
    # Convert seed to numbers
    pattern = [char_to_int[char] for char in seed_text if char in char_to_int]
    
    generated = ""
    
    for _ in range(n_chars):
        # Prepare input
        x = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(x, verbose=0)
        
        # Get the best next character
        index = np.argmax(prediction)
        next_char = int_to_char[index]
        
        generated += next_char
        pattern.append(index)
        pattern = pattern[1:]   # keep only last 50 characters
    
    return seed_text + generated


# -------------------------------- MAIN PROGRAM 
if __name__ == "__main__":
    # Change this if your file name is different
    file_path = "gutenberg.txt"
    
    print("🚀 Starting Text Generator Project...\n")
    
    # Load data
    text = load_text(file_path)
    
    # Prepare sequences
    X, y, char_to_int, int_to_char, vocab_size = prepare_data(text, seq_length=50)
    
    # Create model
    model = create_model(vocab_size, 50)
    
    
    print("Model is ready for text generation!")
    
    # Generate sample text
    seeds = [
        "to be or not to be",
        "hello my dear friend",
        "all the world is"
    ]
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        generated = generate_text(model, seed, char_to_int, int_to_char, n_chars=200)
        print(generated)
        print('='*60)