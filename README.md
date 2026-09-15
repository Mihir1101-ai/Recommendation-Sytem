# 🧠 Next Word Prediction using LSTM

A Deep Learning project that predicts the **next word in a sentence or quote** using a Long Short-Term Memory (LSTM) neural network.

The model is trained on a dataset of quotes and learns the patterns and relationships between words to generate the most likely next word.

---

## 🚀 Features

- 📝 Enter a sentence or quote
- 🔮 Predict the next word
- 🧠 LSTM-based sequence prediction
- 🔤 Tokenization and sequence preprocessing
- 📏 Sequence padding for fixed-length input
- ⚡ TensorFlow/Keras implementation
- 🌐 Streamlit web interface

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Streamlit
- Scikit-learn
- Pickle

---

## 🧠 Model Architecture

The model uses the following architecture:

```text
Input Text
    ↓
Tokenizer
    ↓
Sequence Padding
    ↓
Embedding Layer
    ↓
LSTM Layer
    ↓
Dense Layer
    ↓
Softmax
    ↓
Predicted Next Word
