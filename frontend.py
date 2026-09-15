import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import h5py

# Page Configuration
st.set_page_config(
    page_title="AI Next-Word Recommendation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Premium Aesthetic
st.markdown("""
<style>
    /* Main Theme Overrides */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Gradient Header */
    .header-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-shadow: 0 10px 30px rgba(56, 189, 248, 0.2);
    }

    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Glassmorphic Container Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* Metric Badges */
    .stat-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        color: #a5b4fc;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* Word Pills / Buttons styling */
    .stButton>button {
        border-radius: 12px !important;
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.3) !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.5) !important;
        background: linear-gradient(135deg, #60a5fa 0%, #818cf8 100%) !important;
    }

    /* Output text styling */
    .generated-text-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        font-size: 1.15rem;
        color: #e2e8f0;
        line-height: 1.6;
        min-height: 100px;
    }

    .generated-highlight {
        color: #38bdf8;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# Lightweight Tokenizer replacement for unpickling Keras tokenizer without TensorFlow/Keras
class SimpleTokenizer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            text = text.lower()
            for c in filters:
                text = text.replace(c, ' ')
            words = text.split()
            seq = []
            for w in words:
                idx = self.word_index.get(w)
                if idx is not None:
                    if hasattr(self, 'num_words') and self.num_words and idx >= self.num_words:
                        continue
                    seq.append(idx)
                elif hasattr(self, 'oov_token') and self.oov_token:
                    oov_idx = self.word_index.get(self.oov_token)
                    if oov_idx:
                        seq.append(oov_idx)
            sequences.append(seq)
        return sequences


class TokenizerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Tokenizer':
            return SimpleTokenizer
        return super().find_class(module, name)


# Lightweight pad_sequences implementation
def pad_sequences(sequences, maxlen=None, padding='pre'):
    result = []
    for seq in sequences:
        if len(seq) < maxlen:
            if padding == 'pre':
                padded = [0] * (maxlen - len(seq)) + seq
            else:
                padded = seq + [0] * (maxlen - len(seq))
        else:
            if padding == 'pre':
                padded = seq[-maxlen:]
            else:
                padded = seq[:maxlen]
        result.append(padded)
    return np.array(result)


# Pure NumPy LSTM Inference Engine for next_word_lstm.h5
class NumpyLSTMModel:
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            mw = f['model_weights']
            self.emb_weights = self._find_dataset(mw['embedding_2'])
            
            lstm_grp = mw['lstm_2']
            self.lstm_kernel = self._find_dataset_by_name(lstm_grp, 'kernel')
            self.lstm_rec_kernel = self._find_dataset_by_name(lstm_grp, 'recurrent_kernel')
            self.lstm_bias = self._find_dataset_by_name(lstm_grp, 'bias')
            
            dense_grp = mw['dense_2']
            self.dense_kernel = self._find_dataset_by_name(dense_grp, 'kernel')
            self.dense_bias = self._find_dataset_by_name(dense_grp, 'bias')

    def _find_dataset(self, group):
        for k, v in group.items():
            if isinstance(v, h5py.Dataset):
                return v[:]
            elif isinstance(v, h5py.Group):
                res = self._find_dataset(v)
                if res is not None:
                    return res
        return None

    def _find_dataset_by_name(self, group, name):
        if name in group and isinstance(group[name], h5py.Dataset):
            return group[name][:]
        for k, v in group.items():
            if isinstance(v, h5py.Group):
                res = self._find_dataset_by_name(v, name)
                if res is not None:
                    return res
        return None

    def predict(self, sequence_padded, verbose=0):
        seqs = sequence_padded
        results = []
        units = self.lstm_rec_kernel.shape[0]

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

        for seq in seqs:
            h = np.zeros((units,), dtype=np.float32)
            c = np.zeros((units,), dtype=np.float32)

            for token in seq:
                x_t = self.emb_weights[token]
                z = np.dot(x_t, self.lstm_kernel) + np.dot(h, self.lstm_rec_kernel) + self.lstm_bias
                
                z_i = z[0:units]
                z_f = z[units:2*units]
                z_c = z[2*units:3*units]
                z_o = z[3*units:4*units]

                i_gate = sigmoid(z_i)
                f_gate = sigmoid(z_f)
                c_cand = np.tanh(z_c)
                o_gate = sigmoid(z_o)

                c = f_gate * c + i_gate * c_cand
                h = o_gate * np.tanh(c)

            logits = np.dot(h, self.dense_kernel) + self.dense_bias
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            results.append(probs)

        return np.array(results)


# Cache Model & Artifact Loading for Optimal Performance
@st.cache_resource
def load_model_artifacts():
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = TokenizerUnpickler(f).load()
        
        with open('max_len.pkl', 'rb') as f:
            max_len = pickle.load(f)
            
        model = NumpyLSTMModel('next_word_lstm.h5')
        return tokenizer, max_len, model, None
    except Exception as e:
        return None, None, None, str(e)


# Helper Function: Get Top-K Next Word Predictions
def predict_next_words(model, tokenizer, max_len, text, top_k=5):
    text = text.strip()
    if not text:
        return []
    
    # Tokenize text
    sequence = tokenizer.texts_to_sequences([text])[0]
    if not sequence:
        return []
    
    # Pad sequence to max_len
    padded = pad_sequences([sequence], maxlen=max_len, padding='pre')
    
    # Get model probabilities
    predictions = model.predict(padded, verbose=0)[0]
    
    # Get top_k indices sorted descending
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        word = tokenizer.index_word.get(idx, '<OOV>')
        prob = float(predictions[idx])
        percentage = prob * 100
        if word != '<OOV>' and idx != 0:
            results.append({
                'word': word,
                'probability': prob,
                'percentage': percentage
            })
            
    return results


# Helper Function: Auto Generate Multiple Next Words
def generate_text_sequence(model, tokenizer, max_len, seed_text, num_words=5, temperature=1.0):
    current_text = seed_text.strip()
    generated_words = []
    
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([current_text])[0]
        if not sequence:
            break
        
        padded = pad_sequences([sequence], maxlen=max_len, padding='pre')
        preds = model.predict(padded, verbose=0)[0]
        
        # Temperature scaling
        if temperature > 0 and temperature != 1.0:
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds + 1e-10) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            next_idx = np.random.choice(len(preds), p=preds)
        else:
            next_idx = np.argmax(preds)
            
        next_word = tokenizer.index_word.get(next_idx, '')
        if not next_word or next_word == '<OOV>':
            break
            
        current_text += ' ' + next_word
        generated_words.append(next_word)
        
    return current_text, generated_words


# Main App Layout
def main():
    # Sidebar - Architecture & Stats
    st.sidebar.markdown("### 🧠 Model Architecture")
    st.sidebar.markdown("""
    **RNN / LSTM Model Specs:**
    - **Architecture**: `Embedding` (128) ➔ `LSTM` (128) ➔ `Dense` Softmax
    - **Vocabulary Size**: 6,739 unique tokens
    - **Max Sequence Length**: 30 words
    - **Inference Engine**: Pure NumPy (TensorFlow-Free)
    """)
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 💡 Quick Sample Prompts")
    sample_prompts = [
        "how are you",
        "what is the",
        "i would like to",
        "deep learning is",
        "the recommendation system",
        "we need to"
    ]
    
    # Session state for user input
    if 'user_prompt' not in st.session_state:
        st.session_state.user_prompt = "how are you"
        
    for sample in sample_prompts:
        if st.sidebar.button(f"📌 {sample}", key=f"btn_{sample}", use_container_width=True):
            st.session_state.user_prompt = sample
            st.rerun()
            
    st.sidebar.markdown("---")
    st.sidebar.info("✨ Developed with Deep Learning & Streamlit")

    # Load Model Artifacts
    tokenizer, max_len, model, err = load_model_artifacts()
    
    # Header Banner
    st.markdown('<h1 class="header-title">AI Next-Word Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Real-time word completion and intelligent sentence continuation powered by Recurrent Neural Networks (LSTM).</p>', unsafe_allow_html=True)
    
    if err or model is None:
        st.error(f"❌ Error loading model artifacts: {err}")
        st.stop()
        
    # Top Stats Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stat-badge">📚 Vocabulary: 6,739 Words</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-badge">📏 Max Context: 30 Tokens</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-badge">⚙️ Embedding Dim: 128</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-badge">🔄 Recurrent Units: 128 LSTM</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs Layout
    tab1, tab2 = st.tabs(["🔮 Real-Time Next Word Predictor", "⚡ Auto Text Generation"])
    
    # TAB 1: Next Word Predictor
    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("1. Type Your Sentence Prompt")
        
        # User Text Input
        prompt_input = st.text_input(
            "Enter text to get immediate next-word recommendations:",
            value=st.session_state.user_prompt,
            key="prompt_input_key",
            placeholder="Type a sentence here..."
        )
        
        st.session_state.user_prompt = prompt_input
        
        # Slider for Top-K candidate count
        top_k = st.slider("Number of Top Recommendations:", min_value=3, max_value=10, value=5)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if prompt_input.strip():
            # Get Predictions
            with st.spinner("Analyzing sequence & predicting next word..."):
                predictions = predict_next_words(model, tokenizer, max_len, prompt_input, top_k=top_k)
                
            if predictions:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("2. Recommended Next Words (Click to Append)")
                
                # Render Word Recommendation Buttons
                cols = st.columns(len(predictions))
                for idx, item in enumerate(predictions):
                    with cols[idx]:
                        btn_label = f"➕ **{item['word']}**\n\n({item['percentage']:.1f}%)"
                        if st.button(btn_label, key=f"append_{idx}_{item['word']}", use_container_width=True):
                            # Append word to prompt and rerun
                            st.session_state.user_prompt = (prompt_input.strip() + " " + item['word']).strip()
                            st.rerun()
                            
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Probability Distribution Chart
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("3. Probability Distribution Breakdown")
                
                df_preds = pd.DataFrame(predictions)
                df_preds['Confidence (%)'] = df_preds['percentage'].round(2)
                df_preds.rename(columns={'word': 'Word'}, inplace=True)
                
                st.bar_chart(
                    data=df_preds.set_index('Word')['Confidence (%)'],
                    color="#6366f1",
                    height=250
                )
                
                # Detailed Table view
                with st.expander("📊 View Detailed Confidence Scores Table"):
                    st.dataframe(
                        df_preds[['Word', 'Confidence (%)', 'probability']],
                        use_container_width=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ No valid prediction words found for the input phrase. Try entering common words or phrases.")

    # TAB 2: Multi-Word Auto Generation
    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Automated Sentence Continuation")
        st.write("Let the LSTM model generate a full sequence of upcoming words based on your prompt.")
        
        gen_prompt = st.text_input(
            "Seed Prompt for Generation:",
            value=st.session_state.user_prompt,
            key="gen_prompt_input"
        )
        
        col_gen1, col_gen2 = st.columns(2)
        with col_gen1:
            num_words_to_gen = st.slider("Number of Words to Generate:", min_value=1, max_value=20, value=7)
        with col_gen2:
            temperature = st.slider("Creativity (Temperature):", min_value=0.1, max_value=1.5, value=0.7, step=0.1,
                                    help="Lower temperature values produce deterministic/predictable text. Higher values increase randomness and creativity.")
            
        if st.button("🚀 Generate Sequence", key="btn_generate_text", use_container_width=True):
            if gen_prompt.strip():
                with st.spinner("Generating sequence using LSTM..."):
                    full_text, new_words = generate_text_sequence(
                        model, tokenizer, max_len, gen_prompt, num_words=num_words_to_gen, temperature=temperature
                    )
                    
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Generated Text Output:")
                
                highlighted_new = " ".join([f'<span class="generated-highlight">{w}</span>' for w in new_words])
                output_html = f'<div class="generated-text-box">{gen_prompt} {highlighted_new}</div>'
                
                st.markdown(output_html, unsafe_allow_html=True)
                st.success(f"Generated {len(new_words)} words successfully!")
            else:
                st.warning("Please enter a seed prompt first.")
                
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
