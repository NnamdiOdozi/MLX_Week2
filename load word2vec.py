import wandb
import os
import torch
from dotenv import load_dotenv


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