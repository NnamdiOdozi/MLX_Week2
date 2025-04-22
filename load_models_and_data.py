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

def text_to_embeddings(text, word_to_idx, embeddings, unknown_token_id=0):
    """Convert text to token embeddings"""
    # Tokenize the text
    tokens = preprocess_for_inference(text)
    
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


# Function to calculate cosine similarity between two embeddings
def calc_cosine_sim(emb1, emb2):
    # Handle empty embeddings
    if emb1.shape[0] == 0 or emb2.shape[0] == 0:
        return 0.0
    
    a = emb1.mean(dim=0)
    b = emb2.mean(dim=0)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# Function to process a single row and return similarities
def calculate_similarities(row, word_to_idx, embeddings):
    """Convert text to token embeddings and calculate similarities"""
    # Get embeddings
    query_emb = text_to_embeddings(row['query'], word_to_idx, embeddings)
    pos_emb = text_to_embeddings(row['positive_passage'], word_to_idx, embeddings) 
    neg_emb = text_to_embeddings(row['negative_passage'], word_to_idx, embeddings)
    
    # Calculate average embeddings
    avg_query_emb = query_emb.mean(dim=0).detach().numpy() if query_emb.shape[0] > 0 else np.zeros(embeddings.shape[1])
    avg_pos_emb = pos_emb.mean(dim=0).detach().numpy() if pos_emb.shape[0] > 0 else np.zeros(embeddings.shape[1])
    avg_neg_emb = neg_emb.mean(dim=0).detach().numpy() if neg_emb.shape[0] > 0 else np.zeros(embeddings.shape[1])
    
    # Calculate similarities
    if query_emb.shape[0] > 0 and pos_emb.shape[0] > 0:
        query_pos_sim = F.cosine_similarity(
            query_emb.mean(dim=0).unsqueeze(0), 
            pos_emb.mean(dim=0).unsqueeze(0)
        ).item()
    else:
        query_pos_sim = 0.0
        
    if query_emb.shape[0] > 0 and neg_emb.shape[0] > 0:
        query_neg_sim = F.cosine_similarity(
            query_emb.mean(dim=0).unsqueeze(0), 
            neg_emb.mean(dim=0).unsqueeze(0)
        ).item()
    else:
        query_neg_sim = 0.0
    
    if pos_emb.shape[0] > 0 and neg_emb.shape[0] > 0:
        pos_neg_sim = F.cosine_similarity(
            pos_emb.mean(dim=0).unsqueeze(0), 
            neg_emb.mean(dim=0).unsqueeze(0)
        ).item()
    else:
        pos_neg_sim = 0.0
    
    # Create a Series with both similarities and average embeddings
    result = pd.Series({
        'avg_query_embedding': avg_query_emb,
        'avg_pos_embedding': avg_pos_emb,
        'avg_neg_embedding': avg_neg_emb,
        'query_pos_sim': query_pos_sim,
        'query_neg_sim': query_neg_sim,
        'pos_neg_sim': pos_neg_sim
        
    })
    
    return result







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