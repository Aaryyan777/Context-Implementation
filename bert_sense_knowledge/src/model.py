import torch
from transformers import BertTokenizerFast, BertModel
import numpy as np

class BertSenseEncoder:
    def __init__(self, model_name='bert-base-uncased', device='cpu'):
        print(f"Loading BERT model: {model_name}...")
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, batch_data):
        """
        Extracts embeddings for a batch of occurrences.
        
        Args:
            batch_data: List of dicts, each containing:
                - 'sentence_tokens': list of str
                - 'target_index': int (index of the target word in sentence_tokens)
                
        Returns:
            np.array: Matrix of embeddings (batch_size, embedding_dim)
        """
        embeddings = []
        
        # Process one by one for simplicity (or batch if needed, but alignment is tricky in batch)
        # Given the dataset size (thousands), one-by-one might be slow but safer for alignment correctness.
        # We can optimize to small batches if needed.
        
        for item in batch_data:
            tokens = item['sentence_tokens']
            target_idx = item['target_index']
            
            # Tokenize with offset mapping
            # is_split_into_words=True assumes 'tokens' is already a list of words
            inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True)
            
            # Get word_ids from the BatchEncoding object BEFORE converting to dict
            word_ids = inputs.word_ids(batch_index=0)
            
            # Move tensors to device
            model_inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Filter for the subwords corresponding to target_idx
            # word_ids is a list like [None, 0, 1, 1, 2, None]
            target_subword_indices = [i for i, w_id in enumerate(word_ids) if w_id == target_idx]
            
            if not target_subword_indices:
                # Should not happen if data is consistent, but truncation might cut it off
                # print(f"Warning: Target word at index {target_idx} not found (likely truncated). Skipping.")
                embeddings.append(None)
                continue
                
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            
            # Paper: "summed activations of the final four layers"
            # hidden_states is a tuple of (layer_0, ..., layer_12)
            # We want the last 4: -1, -2, -3, -4
            hidden_states = outputs.hidden_states
            last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
            
            # Stack and sum: (4, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            summed_layers = torch.stack(last_four_layers).sum(dim=0).squeeze(0) # Remove batch dim 0
            
            # Extract embedding for target subwords
            # Strategy: Average the embeddings of the subwords that make up the target word
            # There are other strategies (first subword), but average is common.
            # The paper doesn't explicitly state subword strategy, but "activations corresponding to the type" suggests covering the whole token.
            
            target_vectors = summed_layers[target_subword_indices] # (num_subwords, hidden_dim)
            final_vector = target_vectors.mean(dim=0) # (hidden_dim)
            
            embeddings.append(final_vector.cpu().numpy())
            
        return embeddings
