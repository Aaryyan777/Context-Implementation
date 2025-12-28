import numpy as np
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from collections import Counter
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
import matplotlib.cm as cm

def compute_sense_entropy(senses):
    """
    Computes Shannon entropy of the sense distribution.
    H = - sum (p(s) * log(p(s)))
    """
    counts = Counter(senses)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * np.log(p)
    return entropy

def compute_centroids(embeddings, senses, min_count=10):
    """
    Computes centroids for each sense.
    """
    sense_groups = {}
    for emb, sense in zip(embeddings, senses):
        if sense not in sense_groups:
            sense_groups[sense] = []
        sense_groups[sense].append(emb)
        
    centroids = {}
    valid_senses = []
    
    for sense, embs in sense_groups.items():
        if len(embs) >= min_count:
            centroids[sense] = np.mean(embs, axis=0)
            valid_senses.append(sense)
    
    return centroids, valid_senses

def compute_pairwise_distances(centroids, valid_senses):
    """
    Computes pairwise cosine distances between sense centroids.
    """
    distances = {}
    n = len(valid_senses)
    for i in range(n):
        for j in range(i + 1, n):
            s1 = valid_senses[i]
            s2 = valid_senses[j]
            dist = cosine(centroids[s1], centroids[s2])
            distances[(s1, s2)] = dist
            
    return distances

def run_classification_full(embeddings, senses, min_count=10):
    """
    Runs 5-fold cross-validation logistic regression (L1 penalty) and computes baselines.
    Also returns pairwise confusion if possible.
    
    Returns:
        dict containing:
        - 'f1': Weighted F1 score
        - 'baseline_random': F1 of random classifier
        - 'baseline_majority': F1 of majority class classifier
        - 'confusion_matrix': confusion matrix (normalized) or None
        - 'classes': class labels
    """
    counts = Counter(senses)
    valid_indices = [i for i, s in enumerate(senses) if counts[s] >= min_count]
    
    if len(valid_indices) < 20:
        return None
        
    X = embeddings[valid_indices]
    y = np.array(senses)[valid_indices]
    
    if len(set(y)) < 2:
        return None
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Main Model (Logistic Regression with L1 penalty per paper)
    # Solver 'liblinear' supports L1.
    clf = LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear', multi_class='ovr')
    
    # 2. Baselines
    dummy_majority = DummyClassifier(strategy="most_frequent")
    dummy_random = DummyClassifier(strategy="uniform", random_state=42)
    
    try:
        f1_scores = cross_val_score(clf, X, y, cv=skf, scoring='f1_weighted')
        f1_maj = cross_val_score(dummy_majority, X, y, cv=skf, scoring='f1_weighted').mean()
        f1_rnd = cross_val_score(dummy_random, X, y, cv=skf, scoring='f1_weighted').mean()
        
        # 3. Pairwise Confusion
        # We need predictions to build confusion matrix. 
        y_pred = cross_val_predict(clf, X, y, cv=skf)
        classes = sorted(list(set(y))) # sklearn sorts classes
        cm_norm = confusion_matrix(y, y_pred, labels=classes, normalize='true')
        
        return {
            'f1': f1_scores.mean(),
            'baseline_majority': f1_maj,
            'baseline_random': f1_rnd,
            'confusion_matrix': cm_norm,
            'classes': classes
        }
        
    except Exception as e:
        print(f"Classification failed: {e}")
        return None

def compute_pairwise_confusion_distances(conf_matrix, classes):
    """
    Extracts pairwise confusion "relatedness" (or distance).
    Paper Section 4.3: "normalized each item in the matrix... probability an item was predicted given its true class."
    We use these probabilities to measure relatedness.
    
    Returns:
        dict: {(sense_a, sense_b): probability_confused}
    """
    if conf_matrix is None:
        return {}
        
    relatedness = {}
    n = len(classes)
    for i in range(n):
        for j in range(i + 1, n):
            s1 = classes[i]
            s2 = classes[j]
            # Confusion is not symmetric. Paper summed? "we considered all entries in both matrices"
            # Simple metric: Average confusion (s1 classified as s2 + s2 classified as s1)
            prob_s1_as_s2 = conf_matrix[i, j]
            prob_s2_as_s1 = conf_matrix[j, i]
            
            # This is "relatedness". Higher = closer.
            avg_prob = (prob_s1_as_s2 + prob_s2_as_s1) / 2.0
            relatedness[(s1, s2)] = avg_prob
            
    return relatedness

def plot_dendrogram(centroids, valid_senses, word_name, save_path):
    if len(valid_senses) < 2:
        return
    matrix = np.array([centroids[s] for s in valid_senses])
    Z = linkage(matrix, method='average', metric='cosine')
    plt.figure(figsize=(10, 7))
    dendrogram(Z, labels=valid_senses, orientation='right')
    plt.title(f'Sense Dendrogram for "{word_name}"')
    plt.xlabel('Cosine Distance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved dendrogram to {save_path}")

def plot_tsne(embeddings, senses, word_name, save_path, perplexity=30):
    counts = Counter(senses)
    valid_indices = [i for i, s in enumerate(senses) if counts[s] >= 3]
    if len(valid_indices) < 5:
        return

    X = embeddings[valid_indices]
    y = np.array(senses)[valid_indices]
    unique_senses = list(set(y))
    n_samples = X.shape[0]
    eff_perplexity = min(perplexity, n_samples - 1) if n_samples > 1 else 1

    tsne = TSNE(n_components=2, perplexity=eff_perplexity, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_senses)))
    
    for sense, color in zip(unique_senses, colors):
        indices = np.where(y == sense)
        plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], label=sense, color=color, alpha=0.7)
        
    plt.title(f't-SNE of BERT Embeddings for "{word_name}"')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")

def plot_confusion_matrix(cm, classes, word_name, save_path):
    """
    Plots and saves the confusion matrix as a heatmap.
    """
    if cm is None:
        return

    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Sense Confusion Matrix for "{word_name}"')
    plt.colorbar(im)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Label cells
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Sense')
    plt.xlabel('Predicted Sense')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")