import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'bert_sense_knowledge', 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    import setup
    from src.data_loader import load_semcor_data
    from src.model import BertSenseEncoder
    from src.analysis import (compute_centroids, compute_pairwise_distances, 
                            run_classification_full, compute_pairwise_confusion_distances, 
                            compute_sense_entropy)
except ImportError:
    sys.path.append(os.getcwd())
    import setup
    from src.data_loader import load_semcor_data
    from src.model import BertSenseEncoder
    from src.analysis import (compute_centroids, compute_pairwise_distances, 
                            run_classification_full, compute_pairwise_confusion_distances, 
                            compute_sense_entropy)

st.set_page_config(page_title="BERT Sense Explorer", layout="wide")

@st.cache_resource
def get_model(device_name):
    return BertSenseEncoder(device=device_name)

@st.cache_data
def get_data(words):
    setup.download_nltk_data()
    return load_semcor_data(words)

def plot_dendrogram_figure(centroids, valid_senses):
    if len(valid_senses) < 2:
        return None
    matrix = np.array([centroids[s] for s in valid_senses])
    Z = linkage(matrix, method='average', metric='cosine')
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=valid_senses, orientation='right', ax=ax)
    ax.set_title('Sense Dendrogram (Family Tree of Meanings)')
    ax.set_xlabel('Cosine Distance (Farther = More Different)')
    return fig

def plot_tsne_figure(embeddings, senses):
    from collections import Counter
    counts = Counter(senses)
    valid_indices = [i for i, s in enumerate(senses) if counts[s] >= 3]
    if len(valid_indices) < 5:
        return None

    X = embeddings[valid_indices]
    y = np.array(senses)[valid_indices]
    unique_senses = list(set(y))
    n_samples = X.shape[0]
    eff_perplexity = min(30, n_samples - 1) if n_samples > 1 else 1

    tsne = TSNE(n_components=2, perplexity=eff_perplexity, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_senses)))
    
    for sense, color in zip(unique_senses, colors):
        indices = np.where(y == sense)
        ax.scatter(X_embedded[indices, 0], X_embedded[indices, 1], label=sense, color=color, alpha=0.7)
        
    ax.set_title('t-SNE Map (Clustering of Meanings)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_confusion_figure(cm, classes):
    if cm is None:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Sense Confusion Matrix')
    fig.colorbar(im, ax=ax)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True Sense')
    ax.set_xlabel('Predicted Sense')
    plt.tight_layout()
    return fig

# --- Sidebar ---
st.sidebar.title("Configuration")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.caption(f"Running on: {device.upper()}")

st.sidebar.subheader("Input")
target_word = st.sidebar.text_input("Enter a Word", value="table", help="Type a word to explore its different meanings.")
min_count = st.sidebar.slider("Min Samples per Sense", 3, 20, 5, help="Only analyze senses that appear at least this many times.")

st.sidebar.subheader("Suggested Words")
st.sidebar.markdown("""
Try these to see different patterns:
*   **table** (Distinct meanings: furniture vs. data)
*   **chicken** (Related meanings: animal vs. food)
*   **cover** (Many complex meanings)
*   **bank** (Distinct: money vs. river)
*   **field** (Study vs. land)
""")

run_btn = st.sidebar.button("Run Analysis", type="primary")

# --- Main Page ---
st.title("🧠 BERT Sense Knowledge Explorer")
st.markdown("""
**How does Artificial Intelligence understand words with multiple meanings?**

This tool uses **BERT** (a powerful language model) to "read" sentences from a database and visualize how it understands different word senses.
It distinguishes between **Homonymy** (completely different meanings, like *bat*) and **Polysemy** (related meanings, like *chicken*).
""")

if run_btn:
    with st.spinner(f"Reading thousands of sentences to find '{target_word}'..."):
        # Load Data
        data_map = get_data([target_word])
        items = data_map.get(target_word, [])
        
    if not items:
        st.error(f"❌ No occurrences found for '{target_word}' in the SemCor database.")
        st.info("Try a common word like 'run', 'light', 'table', or 'face'.")
    else:
        st.success(f"✅ Found {len(items)} examples of '{target_word}'.")
        
        with st.spinner("Analyzing BERT's brain..."):
            encoder = get_model(device)
            embeddings_list = encoder.get_embeddings(items)
        
        # Filter
        valid_items = []
        valid_embeddings = []
        valid_senses = []
        
        for item, emb in zip(items, embeddings_list):
            if emb is not None:
                valid_items.append(item)
                valid_embeddings.append(emb)
                valid_senses.append(item['sense'])
        
        valid_embeddings = np.array(valid_embeddings)
        
        # --- Metrics ---
        st.divider()
        st.header("1. Overview")
        
        entropy = compute_sense_entropy(valid_senses)
        centroids, valid_sense_labels = compute_centroids(valid_embeddings, valid_senses, min_count=min_count)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sentences", len(valid_senses))
        col1.caption("Number of examples analyzed")
        
        col2.metric("Distinct Meanings Found", len(valid_sense_labels))
        col2.caption(f"Senses with >={min_count} examples")
        
        col3.metric("Ambiguity Score (Entropy)", f"{entropy:.2f}")
        col3.caption("0 = One meaning. Higher = Many balanced meanings.")

        # --- Interpretation Block ---
        if len(valid_sense_labels) < 2:
            st.warning(f"⚠️ **Monosemy Detected**: BERT mostly found only one dominant meaning for '{target_word}' (`{valid_sense_labels[0] if valid_sense_labels else 'none'}`).")
            st.markdown("Try lowering the **Min Samples** slider to 3 to see rarer meanings, or try a more ambiguous word.")
        else:
            # --- Analysis ---
            st.divider()
            st.header("2. Visualizing Meanings")
            st.markdown("We use math to calculate how 'far apart' different meanings are in BERT's understanding.")

            tab1, tab2, tab3 = st.tabs(["🗺️ Geometry & Maps", "🤖 Can BERT Tell Them Apart?", "📄 Raw Data"])
            
            with tab1:
                col_d, col_t = st.columns(2)
                
                with col_d:
                    st.subheader("Family Tree of Meanings")
                    st.caption("Meanings that are closer together in this tree are more related.")
                    fig_d = plot_dendrogram_figure(centroids, valid_sense_labels)
                    if fig_d: st.pyplot(fig_d)
                    
                    with st.expander("ℹ️ How to read this"):
                        st.markdown("""
                        *   **Horizontal Line Length**: Represents distance. Longer lines mean meanings are very different (Homonyms).
                        *   **Clustering**: Meanings that join together early are synonyms or closely related (Polysemes).
                        """)
                
                with col_t:
                    st.subheader("Meaning Map (t-SNE)")
                    st.caption("Each dot is a sentence. Colors represent different meanings.")
                    fig_t = plot_tsne_figure(valid_embeddings, valid_senses)
                    if fig_t: st.pyplot(fig_t)
                    else: st.info("Not enough data points to generate a map.")

            with tab2:
                st.subheader("AI Classification Test")
                st.markdown("If we train a simple AI to guess the meaning based on BERT's understanding, how well does it do?")
                
                cls_results = run_classification_full(valid_embeddings, valid_senses, min_count=min_count)
                
                if cls_results:
                    score = cls_results['f1']
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Distinctness Score (F1)", f"{score:.2f}")
                    if score > 0.9:
                        c1.success("🌟 Excellent! BERT can perfectly distinguish these meanings. They are likely unrelated (Homonyms).")
                    elif score > 0.7:
                        c1.info("👌 Good. The meanings are distinct but share some context.")
                    else:
                        c1.warning("⚠️ Low. BERT finds these meanings very similar (Polysemes).")

                    c2.metric("Random Guessing", f"{cls_results['baseline_random']:.2f}")
                    c2.caption("Score if we just guessed randomly")
                    
                    st.subheader("Where does BERT get confused?")
                    fig_cm = plot_confusion_figure(cls_results['confusion_matrix'], cls_results['classes'])
                    if fig_cm: st.pyplot(fig_cm)
                else:
                    st.warning("Skipped classification (not enough data).")

            with tab3:
                st.subheader("Example Sentences")
                st.markdown("Here are some of the actual sentences BERT read:")
                # Show first 20 examples
                sample_df = pd.DataFrame([{
                    'Meaning ID': item['sense'],
                    'Sentence': item['sentence_str']
                } for item in valid_items[:20]])
                st.table(sample_df)