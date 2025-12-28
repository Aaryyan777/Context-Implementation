import nltk
from nltk.corpus import semcor
from nltk.tree import Tree
from collections import defaultdict

def load_semcor_data(target_words):
    """
    Scans SemCor for sentences containing specific target words (lemmas).
    
    Args:
        target_words (list of str): List of target lemmas to search for (e.g., ['table', 'cover']).
        
    Returns:
        dict: A dictionary where keys are target words and values are lists of data points.
              Each data point is a dict: {'sentence': str, 'sense': str, 'target_index': int}
    """
    print(f"Loading SemCor data for targets: {target_words}...")
    
    # Ensure lemmatizer is ready if needed, though SemCor usually provides lemmas in the trees.
    # We will iterate through tagged sentences.
    
    data_store = defaultdict(list)
    target_set = set(target_words)
    
    # semcor.tagged_sents(tag='sem') returns sentences where tokens are either strings or Trees.
    # Trees have labels like Lemma('table.n.02').
    
    for sent_idx, sent in enumerate(semcor.tagged_sents(tag='sem')):
        # Reconstruct the sentence string and find indices of targets
        tokens = []
        target_occurrences = [] # List of (index, lemma_obj)
        
        current_idx = 0
        
        for item in sent:
            if isinstance(item, Tree):
                # It's a tagged token/chunk.
                # The label is usually a Lemma object or similar.
                lemma_obj = item.label()
                
                # The item itself is a list of leaves (words). 
                # Usually single word, but can be multi-word expression.
                chunk_words = item.leaves()
                
                # Check if this chunk corresponds to one of our targets
                # The lemma_obj has a .name() method usually returning 'word.pos.num'
                if hasattr(lemma_obj, 'synset'):
                    # It's likely a Lemma object. 
                    # synset().name() gives 'table.n.02'
                    # name() gives 'table.n.02.table' (specific lemma)
                    
                    # We want to match the *lemma name* (e.g. 'table') against our targets.
                    # lemma_obj.synset().name() gives the sense ID.
                    
                    try:
                        synset_name = lemma_obj.synset().name() # e.g. 'table.n.02'
                        lemma_name = synset_name.split('.')[0] # e.g. 'table'
                        
                        if lemma_name in target_set:
                            # Found a target!
                            # Note: The target might be the first word of the chunk if multi-word.
                            # For BERT alignment, simple approach: take the index of the first token in this chunk.
                            target_occurrences.append({
                                'index': current_idx,
                                'sense': synset_name,
                                'lemma': lemma_name,
                                'original_text': " ".join(chunk_words)
                            })
                    except:
                        pass
                
                tokens.extend(chunk_words)
                current_idx += len(chunk_words)
            elif isinstance(item, list):
                # It's an untagged token represented as a list, e.g. ['the']
                tokens.extend(item)
                current_idx += len(item)
            elif isinstance(item, str):
                # It's a plain string (rare in this corpus reader but possible)
                tokens.append(item)
                current_idx += 1
            else:
                 # Fallback
                 tokens.append(str(item))
                 current_idx += 1
        
        # If we found targets in this sentence, store them
        if target_occurrences:
            sentence_str = " ".join(tokens)
            for occ in target_occurrences:
                data_store[occ['lemma']].append({
                    'sentence_tokens': tokens, # Keep as tokens to help with manual alignment if needed, or join later
                    'sentence_str': sentence_str,
                    'sense': occ['sense'],
                    'target_index': occ['index'],
                    'target_text': occ['original_text']
                })
                
    print(f"Data loading complete.")
    for w in target_words:
        print(f"  Found {len(data_store[w])} occurrences for '{w}'")
        
    return data_store
