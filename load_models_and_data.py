import wandb
import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
import numpy as np
import pandas as pd
import transformers
import torch.nn.functional as F
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import rank_bm25
import torch.nn as nn

# Initialize wandb and login (if not already logged in)
# wandb.login()  # Uncomment if you need to login


# Load environment variables from config.txt
def load_config(config_path="config.txt"):
    # Load the .env file
    load_dotenv(config_path)
    
    # Get the WANDB_API_KEY from environment variables
    api_key = os.getenv("WANDB_API_KEY")
    
    if api_key:
        print("API key loaded successfully")
        return True
    else:
        print("WANDB_API_KEY not found in config file")
        return False

# Set up wandb with the API key from config file
if load_config():
    # Login to wandb (will use the API key from environment variable)
    wandb.login()
else:
    print("Failed to load API key, please check your config.txt file")
    exit(1)

# Download artifacts
def download_model_artifacts():
    # Initialize a run to access artifacts
    api = wandb.Api()
    
    # Define the artifact path in the format "entity/project/artifact_name:version"
    artifact_path = "nnamdi-odozi-ave-actuaries/mlx7-week1-cbow/model-weights:v4"
    
    # Download the artifact
    artifact = api.artifact(artifact_path)
    download_dir = "./downloaded_model"
    artifact_dir = artifact.download(root=download_dir)
    
    print(f"Artifact downloaded to: {artifact_dir}")
    
    # List the files in the downloaded directory
    print("Downloaded files:")
    for f in os.listdir(artifact_dir):
        print(f"- {f}")
    
    return artifact_dir


def preprocess_for_inference(text):
    """Modified preprocessing function that doesn't filter by frequency"""
    text = text.lower()
    text = text.replace('.',  ' <PERIOD> ')
    text = text.replace(',',  ' <COMMA> ')
    text = text.replace('"',  ' <QUOTATION_MARK> ')
    text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace('(',  ' <LEFT_PAREN> ')
    text = text.replace(')',  ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace(':',  ' <COLON> ')
    words = text.split()
    # Remove the frequency filtering step that was in the original preprocess
    return words

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

def text_to_embeddings(text, word_to_idx, embeddings, unknown_token_id=0, is_query=False):
    """
    Convert text to token embeddings with padding and OOV handling
    
    Args:
        text: Input text to convert
        word_to_idx: Dictionary mapping words to indices
        embeddings: Embedding matrix
        unknown_token_id: ID for unknown tokens
        is_query: Flag to determine max length (query vs document)
        
    Returns:
        padded_embeddings: Padded embedding tensor
        length: Actual length of sequence before padding
    """
    
    # Define constants
    MAX_QUERY_LENGTH = 26
    MAX_DOCUMENT_LENGTH = 201
    embedding_dim = 100  # Assuming 100-dim embeddings, modify as needed
    
    # Tokenize the text
    tokens = preprocess_for_inference(text)
    
    # Set max length based on whether this is a query or document
    max_length = MAX_QUERY_LENGTH if is_query else MAX_DOCUMENT_LENGTH
    
    # Convert tokens to indices and create embeddings list
    token_embeddings_list = []
    for token in tokens:
        # Get the token ID or use unknown token ID if not in vocabulary
        idx = word_to_idx.get(token, unknown_token_id)
        
        if idx == unknown_token_id:
            # Create a new random OOV vector for each unknown word
            oov_vector = torch.randn(embedding_dim) * 0.1  # Random but small vector
            token_embeddings_list.append(oov_vector)
        else:
            # Use the regular embedding
            token_embeddings_list.append(embeddings[idx])
    
    # Stack the embeddings (if we have any)
    if token_embeddings_list:
        token_embeddings = torch.stack(token_embeddings_list)
    else:
        # Create an empty tensor with the correct shape if no tokens
        token_embeddings = torch.zeros((0, embedding_dim))
    
    # Get sequence length
    length = len(token_embeddings)
    
    # Create padded tensor initialized with zeros
    padded_embeddings = torch.zeros(max_length, embedding_dim)
    
    # Copy the embeddings to the padded tensor (only up to max_length)
    if length > 0:
        # Only take up to max_length tokens to handle cases where text is too long
        copy_length = min(length, max_length)
        padded_embeddings[:copy_length] = token_embeddings[:copy_length]
    
    return padded_embeddings, length


# Function to calculate cosine similarity between two embeddings
def calc_cosine_sim(emb1, emb2):
    # Handle empty embeddings
    if emb1.shape[0] == 0 or emb2.shape[0] == 0:
        return 0.0
    
    a = emb1.mean(dim=0)
    b = emb2.mean(dim=0)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# Function to process a single row and return embeddings with lengths
def calculate_embeddings(row, word_to_idx, embeddings):
    """Convert text to token embeddings with padding and return lengths"""
    # Get embeddings and lengths
    query_emb, query_length = text_to_embeddings(
        row['query'], word_to_idx, embeddings, is_query=True
    )
    pos_emb, pos_length = text_to_embeddings(
        row['positive_passage'], word_to_idx, embeddings, is_query=False
    ) 
    neg_emb, neg_length = text_to_embeddings(
        row['negative_passage'], word_to_idx, embeddings, is_query=False
    )



    # Calculate average embeddings
    #avg_query_emb = query_emb.mean(dim=0).detach().numpy() if query_emb.shape[0] > 0 else np.zeros(embeddings.shape[1])
    #avg_pos_emb = pos_emb.mean(dim=0).detach().numpy() if pos_emb.shape[0] > 0 else np.zeros(embeddings.shape[1])
    #avg_neg_emb = neg_emb.mean(dim=0).detach().numpy() if neg_emb.shape[0] > 0 else np.zeros(embeddings.shape[1])
    
    # Calculate similarities
    # if query_emb.shape[0] > 0 and pos_emb.shape[0] > 0:
    #     query_pos_sim = F.cosine_similarity(
    #         query_emb.mean(dim=0).unsqueeze(0), 
    #         pos_emb.mean(dim=0).unsqueeze(0)
    #     ).item()
    # else:
    #     query_pos_sim = 0.0
        
    # if query_emb.shape[0] > 0 and neg_emb.shape[0] > 0:
    #     query_neg_sim = F.cosine_similarity(
    #         query_emb.mean(dim=0).unsqueeze(0), 
    #         neg_emb.mean(dim=0).unsqueeze(0)
    #     ).item()
    # else:
    #     query_neg_sim = 0.0
    
    # if pos_emb.shape[0] > 0 and neg_emb.shape[0] > 0:
    #     pos_neg_sim = F.cosine_similarity(
    #         pos_emb.mean(dim=0).unsqueeze(0), 
    #         neg_emb.mean(dim=0).unsqueeze(0)
    #     ).item()
    # else:
    #     pos_neg_sim = 0.0
    
     # Create a Series with embeddings and lengths
    result = pd.Series({
        'query_emb': query_emb,
        'query_length': query_length,
        'pos_emb': pos_emb,
        'pos_length': pos_length,
        'neg_emb': neg_emb,
        'neg_length': neg_length,
    })
    
    return result


def create_packed_batch(embeddings_df):
    """
    Creates packed sequences for batched processing in an RNN
    
    Args:
        embeddings_df: DataFrame containing query_emb, pos_emb, neg_emb and their lengths
        
    Returns:
        packed_queries: PackedSequence for queries
        packed_positives: PackedSequence for positive documents
        packed_negatives: PackedSequence for negative documents
    """
    # Stack all embeddings in the batch
    queries = torch.stack(embeddings_df['query_emb'].tolist())
    positives = torch.stack(embeddings_df['pos_emb'].tolist())
    negatives = torch.stack(embeddings_df['neg_emb'].tolist())
    
    # Get lengths
    query_lengths = torch.tensor(embeddings_df['query_length'].tolist())
    pos_lengths = torch.tensor(embeddings_df['pos_length'].tolist())
    neg_lengths = torch.tensor(embeddings_df['neg_length'].tolist())
    
    # Pack sequences
    packed_queries = nn.utils.rnn.pack_padded_sequence(
        queries, query_lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    packed_positives = nn.utils.rnn.pack_padded_sequence(
        positives, pos_lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    packed_negatives = nn.utils.rnn.pack_padded_sequence(
        negatives, neg_lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    
    return packed_queries, packed_positives, packed_negatives


def main():
    # Download the model artifacts
    artifact_dir = download_model_artifacts()

    # Load the model embeddings and word-to-index mapping
    embeddings_path = os.path.join(artifact_dir, "embeddings.pt")  # Adjust filename if different
    word_to_idx_path = os.path.join(artifact_dir, "word_to_idx.pt")  # Adjust filename if different

    # Load the tensors
    if os.path.exists(embeddings_path):
        embeddings = torch.load(embeddings_path)
        print(f"Embeddings loaded, shape: {embeddings.shape}")
    else:
        print(f"Embeddings file not found at {embeddings_path}")

    if os.path.exists(word_to_idx_path):
        word_to_idx = torch.load(word_to_idx_path)
        print(f"Word to index mapping loaded, vocabulary size: {len(word_to_idx)}")
    else:
        print(f"Word to index mapping file not found at {word_to_idx_path}")

    # Extracting the Embedding layer from the model
    # Load the model
    model_path = "./downloaded_model/2025_04_18__14_41_55.5.cbow.pth"
    model_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Extract the embeddings
    embeddings = model_dict["emb.weight"]

    # Save just the embeddings for future use
    torch.save(embeddings, "./downloaded_model/embeddings.pt")
    print(f"Embeddings saved, shape: {embeddings.shape}")


if __name__ == "__main__":
    main()