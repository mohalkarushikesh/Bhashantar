# Bhashantar : Neural Machine Translation Engine 🌉

An advanced **Sequence-to-Sequence (Seq2Seq)** translation system built with **TensorFlow** and **Keras**, capable of translating between **English** and **Marathi** using **Bidirectional LSTMs**.

## 📌 Project Overview

Unlike traditional rule-based translators, this project implements a Neural Machine Translation (NMT) approach. By using **Bidirectional LSTMs (Long Short-Term Memory)**, the model captures the nuances of Marathi grammar (SOV - Subject-Object-Verb) and English grammar (SVO - Subject-Verb-Object) by reading sequences in both forward and backward directions.

## 🚀 Key Features

* **Dual-Directional Support:** Dedicated training modes for English ↔ Marathi.
* **Bi-LSTM Encoder:** Captures deep contextual relationships from both ends of a sentence.
* **Dynamic Inference:** A standalone inference model for real-time, interactive translation.
* **Smart Preprocessing:** Custom tokenization, padding, and automated handling of `start`/`end` sequence markers.
* **Robust Training:** Equipped with `EarlyStopping` and `ModelCheckpoint` to prevent overfitting and save the best version of the model.

## 🛠️ Technical Stack

* **Language:** Python
* **Deep Learning:** TensorFlow 2.x / Keras 3
* **Data Manipulation:** Pandas, NumPy
* **Serialization:** Pickle (for tokenizers)
* **Dataset:** Bilingual sentence pairs (`mar.txt`)

## 🧠 Model Architecture

1. **Encoder:** Uses a **Bidirectional LSTM** to process the source sentence. It outputs hidden states that represent the "meaning" of the input.
2. **Latent Space:** A 256-dimensional vector space where language context is stored.
3. **Decoder:** An **LSTM** that uses the encoder's final states as its initial states to generate the target sentence word-by-word.

## 📂 File Structure

```text
├── main.py                # The core logic (Training + Inference)
├── mar.txt                # Bilingual dataset
├── eng_to_mar_model.keras # Trained model (generated after training)
├── eng_tokenizer.pkl      # English vocabulary mapping
└── mar_tokenizer.pkl      # Marathi vocabulary mapping

```

## ⚙️ How to Use

### 1. Preparation

Ensure you have the `mar.txt` dataset in the project root. Install dependencies:

```bash
pip install tensorflow pandas numpy

```

### 2. Training

Set the following variables in `main.py`:

```python
TRAIN_MODE = True
MODE = 'E2M'  # Or 'M2E' for Marathi to English

```

Run the script: `python main.py`. This will save your model and tokenizers.

### 3. Translation (Inference)

Once trained, set:

```python
TRAIN_MODE = False

```

Run the script again to enter the **Interactive Translation Loop**.

## 📊 Future Improvements

* **Attention Mechanism:** To handle much longer and more complex sentences.
* **Beam Search:** To improve word selection during the decoding phase.
* **Larger Dataset:** Scaling beyond 15,000 samples for higher fluency.
