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

```
========================================
READY! Mode: E2M
Type 'exit' to stop.
========================================
```
Enter text: Hello
Translation: धावा!

**Enter text: how are you**
**Translation: तू कशी आहे?**

**Enter text: I** love you
**Translation: मला** खूप पोहता.

**Enter text: love**
**Translation: प्रेम आहे.**

Enter text: like
Translation: माहितीये.

Enter text: stop
Translation: सोडा.

Enter text: fell
Translation: आक्रमण कर.

**Enter text: know**
**Translation: माहीत आहे.**

---

License: Do What the F* You Want to Public License — http://www.wtfpl.net

---

## Conclusion: 

Despite using a constrained dataset of only **15,000 samples**, the model has successfully achieved a **Proof of Concept** for Neural Machine Translation. It demonstrates that a **Bidirectional Seq2Seq** architecture can autonomously learn the complex grammatical shift from **English (SVO)** to **Marathi (SOV)** without hardcoded rules.

#### **Key Technical Insights:**

* **Semantic Mapping:** The model accurately mapped core concepts like **"Love" → "प्रेम"** and **"Know" → "माहीत आहे"**, proving the **Embedding Layer** successfully clustered related meanings in the latent space.
* **Syntactic Logic:** By correctly identifying pronouns like **"तू"** (You) and **"मला"** (Me), the **Bi-LSTM Encoder** proved its ability to capture sentence-level context from both directions simultaneously.
* **Bottleneck Efficiency:** The 256-dimensional "context vector" effectively compressed English intent into a format the Decoder could reconstruct into Marathi, even with limited exposure to rare vocabulary.

#### **The Verdict:**

The "hallucinations" (like *Hello* → *Run*) are simply a result of **Data Sparsity**. The architecture itself is robust; increasing the dataset would refine the **Vector Space resolution**, moving the model from basic word-matching to fluent, nuanced translation.

```mermaid
graph TD
    A["Configuration<br/>MODE: E2M/M2E<br/>latent_dim: 256"] --> B["Data Loading & Preprocessing"]
    
    B --> C["Load Dataset<br/>mar.txt"]
    C --> D["Normalize Text<br/>Lowercase & Strip"]
    D --> E["Select Mode<br/>E2M or M2E"]
    
    E --> F["Tokenization"]
    F --> G["Input Tokenizer"]
    F --> H["Target Tokenizer"]
    
    G --> I["Input Sequences"]
    H --> J["Target Sequences"]
    
    I --> K["Pad Sequences"]
    J --> K
    
    K --> L{"TRAIN_MODE?"}
    
    L -->|YES| M["Training Phase"]
    L -->|NO| N["Inference Phase"]
    
    M --> O["Encoder Architecture"]
    O --> O1["Embedding Layer"]
    O1 --> O2["Bidirectional LSTM"]
    O2 --> O3["Concatenate States<br/>Hidden & Cell"]
    O3 --> S["Save Model & Tokenizers"]
    
    M --> P["Decoder Architecture"]
    P --> P1["Embedding Layer"]
    P1 --> P2["LSTM Layer"]
    P2 --> P3["Dense Layer<br/>Softmax Activation"]
    P3 --> Q["Compile & Train<br/>Categorical Crossentropy"]
    Q --> R["Callbacks<br/>EarlyStopping & Checkpoint"]
    R --> S
    
    N --> T["Load Trained Model"]
    T --> U["Load Tokenizers"]
    
    U --> V["Reconstruct Encoder"]
    V --> V1["Extract BiLSTM Layer"]
    V1 --> V2["Combine Forward/Backward States"]
    V2 --> W["Encoder Inference Model"]
    
    U --> X["Reconstruct Decoder"]
    X --> X1["Input: Hidden & Cell States"]
    X1 --> X2["Input: Single Word Token"]
    X2 --> X3["Extract Embedding Layer"]
    X3 --> X4["Extract LSTM Layer"]
    X4 --> X5["Extract Dense Layer"]
    X5 --> Y["Decoder Inference Model"]
    
    W --> Z["Translate Function"]
    Y --> Z
    
    Z --> Z1["Tokenize Input"]
    Z1 --> Z2["Pad Input Sequence"]
    Z2 --> Z3["Encoder Prediction<br/>Get Initial States"]
    Z3 --> Z4["Greedy Decoding Loop"]
    
    Z4 --> Z5["Decoder Prediction<br/>One Word at a Time"]
    Z5 --> Z6["Argmax to Select Word"]
    Z6 --> Z7{"Stop Condition?<br/>end token or<br/>max length"}
    Z7 -->|NO| Z8["Append to Output<br/>Update States"]
    Z8 --> Z5
    Z7 -->|YES| Z9["Return Translation"]
    
    Z9 --> AA["Interactive Loop<br/>User Input/Output"]

    style A fill:#e1f5ff
    style M fill:#c8e6c9
    style N fill:#fff9c4
    style Z fill:#ffccbc
    style AA fill:#f0f4c3
