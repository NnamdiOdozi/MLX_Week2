import torch
import pandas as pd
import numpy as np
from tokenizer import preprocess, preprocess_for_inference # Import your existing tokenizer's preprocess function



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



if __name__ == "__load_models_and_data__":
    main()
    
    

