import sys
import os
import numpy as np
import setup

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data_loader import load_semcor_data
from src.model import BertSenseEncoder
from src.analysis import (compute_centroids, compute_pairwise_distances, 
                            run_classification_full, compute_pairwise_confusion_distances, 
                            compute_sense_entropy, plot_dendrogram, plot_tsne,
                            plot_confusion_matrix)

def main():
    # 1. Ensure data is present
    setup.download_nltk_data()
    
    # 2. Define target words
    target_words = ['table', 'bat', 'chicken', 'cover', 'bank'] 
    
    # 3. Load Data
    data_map = load_semcor_data(target_words)
    
    # 4. Initialize Model
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    encoder = BertSenseEncoder(device=device)
    
    print("\n--- Starting Analysis ---")
    
    for word in target_words:
        items = data_map.get(word, [])
        if not items:
            print(f"No data found for '{word}'. Skipping.")
            continue
            
        print(f"\nProcessing '{word}' ({len(items)} occurrences)...")
        
        # Extract Embeddings
        embeddings_list = encoder.get_embeddings(items)
        
        # Filter out failed extractions (None)
        valid_items = []
        valid_embeddings = []
        valid_senses = []
        
        for item, emb in zip(items, embeddings_list):
            if emb is not None:
                valid_items.append(item)
                valid_embeddings.append(emb)
                valid_senses.append(item['sense'])
                
        if not valid_embeddings:
            print("No valid embeddings extracted.")
            continue
            
        valid_embeddings = np.array(valid_embeddings)
        
        # Metrics: Entropy
        entropy = compute_sense_entropy(valid_senses)
        print(f"Sense Entropy: {entropy:.4f}")

        # Analysis 1: Centroids & Distances
        centroids, valid_sense_labels = compute_centroids(valid_embeddings, valid_senses, min_count=5)
        
        if len(valid_sense_labels) < 2:
            print(f"Not enough senses with sufficient data for '{word}'. Found: {valid_sense_labels}")
            continue
            
        distances = compute_pairwise_distances(centroids, valid_sense_labels)
        
        print("Pairwise Cosine Distances (1 - Similarity):")
        sorted_dists = sorted(distances.items(), key=lambda x: x[1], reverse=True)
        for (s1, s2), dist in sorted_dists:
            print(f"  {s1} vs {s2}: {dist:.4f}")
            
        # Analysis 2: Classification (Full)
        cls_results = run_classification_full(valid_embeddings, valid_senses, min_count=5)
        
        if cls_results:
            print(f"Sense Classification F1 Score (L1 penalty): {cls_results['f1']:.4f}")
            print(f"  Baseline (Majority): {cls_results['baseline_majority']:.4f}")
            print(f"  Baseline (Random):   {cls_results['baseline_random']:.4f}")
            
            # Confusion Relatedness
            conf_relatedness = compute_pairwise_confusion_distances(cls_results['confusion_matrix'], cls_results['classes'])
            if conf_relatedness:
                print("Pairwise Confusion Relatedness (Avg Prob of Confusion):")
                sorted_conf = sorted(conf_relatedness.items(), key=lambda x: x[1], reverse=True)
                # Show top 5 for brevity
                for (s1, s2), prob in sorted_conf[:5]:
                    print(f"  {s1} vs {s2}: {prob:.4f}")
            
            # Visualization: Confusion Matrix Heatmap
            plot_confusion_matrix(cls_results['confusion_matrix'], cls_results['classes'], word, f"{word}_confusion.png")
        else:
            print("Classification skipped (insufficient classes/data).")
            
        # Visualization
        plot_dendrogram(centroids, valid_sense_labels, word, f"{word}_dendrogram.png")
        plot_tsne(valid_embeddings, valid_senses, word, f"{word}_tsne.png")

    print("\n--- Analysis Complete ---")
    print("Visualizations saved as png files.")

if __name__ == "__main__":
    main()
