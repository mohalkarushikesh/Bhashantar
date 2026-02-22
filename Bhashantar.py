import numpy as np
import pandas as pd
import pickle   # serialize (save) trained models & deserialize (load) them back
import os
import tensorflow as tf 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================
# 1. CONFIGURATION
# ==========================================
TRAIN_MODE = False   # Set to TRUE to train, FALSE to just use the translator
MODE = 'E2M'        # 'E2M' for English->Marathi | 'M2E' for Marathi->English
latent_dim = 256
dataset_size = 15000 

if MODE == 'E2M':
    model_name = 'eng_to_mar_model.keras'
    in_tk_name = 'eng_tokenizer.pkl'
    out_tk_name = 'mar_tokenizer.pkl'
else:
    model_name = 'mar_to_eng_model.keras'
    in_tk_name = 'mar_tokenizer.pkl'
    out_tk_name = 'eng_tokenizer.pkl'

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
print(f"Loading dataset for {MODE}...")   # Print which translation mode is active
try:
    lines = pd.read_table('mar.txt', names=['eng', 'mar', 'c'])   # Load dataset with 3 columns
except FileNotFoundError:
    print("CRITICAL ERROR: 'mar.txt' not found!")   # Handle missing file
    exit()

lines = lines[0:dataset_size]   # Limit dataset size for training

# Normalization is the process of rescaling numerical feature values to a common range, typically [0, 1] or [-1, 1], to ensure that all features contribute proportionally during model training.
lines['eng'] = lines['eng'].apply(lambda x: str(x).lower().strip())   # Normalize English text
lines['mar'] = lines['mar'].apply(lambda x: str(x).lower().strip())   # Normalize Marathi text

if MODE == 'E2M':
    input_texts = lines['eng'].values   # Input = English sentences
    target_texts = lines['mar'].apply(lambda x: 'start ' + x + ' end').values   # Target = Marathi with start/end tokens
else:
    input_texts = lines['mar'].values   # Input = Marathi sentences
    target_texts = lines['eng'].apply(lambda x: 'start ' + x + ' end').values   # Target = English with start/end tokens

# Tokenizer helper function
def get_sequences(texts, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer(filters='')   # Create tokenizer without filtering characters
        tokenizer.fit_on_texts(texts)       # Build vocabulary from texts
    return tokenizer, tokenizer.texts_to_sequences(texts)   # Return tokenizer and integer sequences

input_tokenizer, input_seq = get_sequences(input_texts)   # Tokenize input sentences
target_tokenizer, target_seq = get_sequences(target_texts) # Tokenize target sentences

max_in_len = max(len(s) for s in input_seq)   # Find longest input sequence length
max_out_len = max(len(s) for s in target_seq) # Find longest target sequence length

# ==========================================
# 3. TRAINING PHASE
# ==========================================
if TRAIN_MODE:
    print(f"--- STARTING TRAINING ---")
    
    # Padding = fillers (zeros) added to shorter sequences. Purpose = uniform length for efficient batch training.
    # Pad input and target sequences to uniform length
    encoder_input_data = pad_sequences(input_seq, maxlen=max_in_len, padding='post')
    decoder_input_data = pad_sequences(target_seq, maxlen=max_out_len, padding='post')

    # Prepare decoder target data (one-hot encoded for each word)
    decoder_target_data = np.zeros((len(input_texts), max_out_len, len(target_tokenizer.word_index)+1), dtype='float32')
    for i, seq in enumerate(target_seq):
        for t, word_id in enumerate(seq):
            if t > 0: 
                decoder_target_data[i, t - 1, word_id] = 1.0   # Shifted by one timestep

    # Encoder architecture
    encoder_inputs = Input(shape=(max_in_len,), name="encoder_inputs")
    enc_emb = Embedding(len(input_tokenizer.word_index)+1, latent_dim, name="encoder_embedding")(encoder_inputs)
    encoder_bilstm = Bidirectional(LSTM(latent_dim, return_state=True), name="encoder_bilstm")
    _, fh, fc, bh, bc = encoder_bilstm(enc_emb)   # Forward & backward hidden/cell states
    state_h = Concatenate(name="state_h")([fh, bh])   # Combine hidden states
    state_c = Concatenate(name="state_c")([fc, bc])   # Combine cell states
    encoder_states = [state_h, state_c]

    # Decoder architecture
    decoder_inputs = Input(shape=(max_out_len,), name="decoder_inputs")
    dec_emb_layer = Embedding(len(target_tokenizer.word_index)+1, latent_dim, name="decoder_embedding")
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, name="decoder_lstm")
    
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(len(target_tokenizer.word_index)+1, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Compile model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Callbacks: early stopping & best model checkpoint
    callbacks = [
        EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(model_name, monitor='loss', save_best_only=True)
    ]

    # Train the model
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, 
              batch_size=64, 
              epochs=30, 
              callbacks=callbacks)

    # Save tokenizers for inference use
    with open(in_tk_name, 'wb') as f: pickle.dump(input_tokenizer, f)
    with open(out_tk_name, 'wb') as f: pickle.dump(target_tokenizer, f)
    print(f"DONE. Model saved as {model_name}")

# ==========================================
# 4. INFERENCE PHASE 
# ==========================================
print("\n--- INITIALIZING TRANSLATOR ENGINE ---")

# Load trained model and saved tokenizers
trained_model = load_model(model_name)
with open(in_tk_name, 'rb') as f: input_tokenizer = pickle.load(f)
with open(out_tk_name, 'rb') as f: target_tokenizer = pickle.load(f)

# 1. Reconstruct Encoder
enc_inputs = trained_model.inputs[0]   # Encoder input layer
enc_bilstm_layer = trained_model.get_layer("encoder_bilstm")   # Bi-LSTM layer
enc_outputs = enc_bilstm_layer.output   # Outputs from Bi-LSTM

# Extract forward/backward hidden & cell states
fh, fc, bh, bc = enc_outputs[1], enc_outputs[2], enc_outputs[3], enc_outputs[4]
state_h_inf = Concatenate()([fh, bh])   # Combine hidden states
state_c_inf = Concatenate()([fc, bc])   # Combine cell states

encoder_model = Model(inputs=enc_inputs, outputs=[state_h_inf, state_c_inf])   # Encoder inference model

# 2. Reconstruct Decoder
inf_decoder_state_input_h = Input(shape=(latent_dim * 2,), name="inf_dec_h")   # Decoder hidden state input
inf_decoder_state_input_c = Input(shape=(latent_dim * 2,), name="inf_dec_c")   # Decoder cell state input
inf_decoder_states_inputs = [inf_decoder_state_input_h, inf_decoder_state_input_c]

inf_decoder_inputs = Input(shape=(1,), name="inf_dec_word_input")   # One word input at a time

# Use trained decoder layers
inf_dec_emb = trained_model.get_layer("decoder_embedding")(inf_decoder_inputs)
inf_dec_lstm = trained_model.get_layer("decoder_lstm")
inf_dec_outputs, inf_sh, inf_sc = inf_dec_lstm(inf_dec_emb, initial_state=inf_decoder_states_inputs)
inf_dec_dense = trained_model.get_layer("decoder_dense")
inf_dec_outputs = inf_dec_dense(inf_dec_outputs)

decoder_model = Model(
    inputs=[inf_decoder_inputs] + inf_decoder_states_inputs, 
    outputs=[inf_dec_outputs] + [inf_sh, inf_sc]
)

# Translation function
def translate(text):
    seq = input_tokenizer.texts_to_sequences([text.lower().strip()])   # Convert input text to sequence
    seq = pad_sequences(seq, maxlen=max_in_len, padding='post')        # Pad to max length
    
    states_value = encoder_model.predict(seq, verbose=0)   # Get encoder states
    
    target_seq = np.zeros((1, 1))   # Initialize target sequence
    target_seq[0, 0] = target_tokenizer.word_index['start']   # Start token
    
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])   # Pick highest probability word
        sampled_word = target_tokenizer.index_word.get(sampled_token_index, '')
        
        if sampled_word == 'end' or len(decoded_sentence.split()) > 20:   # Stop if 'end' or max length
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word   # Append word to output
            target_seq[0, 0] = sampled_token_index   # Update target sequence
            states_value = [h, c]   # Update decoder states
            
    return decoded_sentence.strip()   # Return final translated sentence

# Interactive loop
print("\n" + "="*40)
print(f"READY! Mode: {MODE}")
print("Type 'exit' to stop.")
print("="*40)

while True:
    user_input = input("\nEnter text: ")
    if user_input.lower() in ['exit', 'quit']: break
    if not user_input.strip(): continue
    
    try:
        print(f"Translation: {translate(user_input)}")
    except Exception as e:
        print(f"Translation Error: {e}")
