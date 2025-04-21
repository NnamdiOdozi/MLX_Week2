import torch
import pandas as pd
import numpy as np
from tokenizer import preprocess  # Import your existing tokenizer's preprocess function

def load_vocabulary(vocab_path):
    """Load the vocabulary mapping from CSV file"""
    vocab_df = pd.read_csv(vocab_path)
    # Create a dictionary mapping words to their token IDs
    word_to_idx = {row['Word']: row['Token_ID'] for _, row in vocab_df.iterrows()}
    return word_to_idx

def load_embeddings(embeddings_path):
    """Load the pre-trained embeddings"""
    embeddings = torch.load(embeddings_path)
    return embeddings

def text_to_embeddings(text, word_to_idx, embeddings, unknown_token_id=0):
    """Convert text to token embeddings"""
    # Tokenize the text
    tokens = preprocess(text)
    
    # Convert tokens to indices
    indices = []
    for token in tokens:
        # Get the token ID or use unknown token ID if not in vocabulary
        idx = word_to_idx.get(token, unknown_token_id)
        indices.append(idx)
    
    # Convert to tensor
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    
    # Look up embeddings
    token_embeddings = embeddings[indices_tensor]
    
    return token_embeddings

def main():
    # Paths to your files
    embeddings_path = "./downloaded_model/embeddings.pt"
    vocab_path = "./downloaded_model/tkn_ids_to_words.csv"
    
    # Load embeddings and vocabulary
    print("Loading embeddings and vocabulary...")
    embeddings = load_embeddings(embeddings_path)
    word_to_idx = load_vocabulary(vocab_path)
    
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    print(f"Loaded vocabulary with {len(word_to_idx)} tokens")
    
    # Example usage (uncomment when ready to test)
    sample_text = "This is a test sentence"
    embeddings_result = text_to_embeddings(sample_text, word_to_idx, embeddings)
    print(f"Embedded text shape: {embeddings_result.shape}")

    # Set numpy print options
    np.set_printoptions(precision=4, suppress=True, threshold=10)  # threshold limits number of elements shown
    numpy_array = embeddings_result.detach().numpy()
    print("Embedding array with custom formatting:")
    print(numpy_array)

    # Your teammate can call text_to_embeddings for each column in the dataframe

if __name__ == "__main__":
    main()