import wandb
import os
import sys
import glob

def upload_model_to_wandb(model_dir):
    """Upload model files from the specified directory to Weights & Biases"""
    # Initialize a new W&B run
    wandb.init(project="twintowers_finetuning", name="twintowers-search")
    
    # Look for model files
    word2vec_file = os.path.join(model_dir, "word2vec_text8_tiny_improved.txt")
    predictor_file = os.path.join(model_dir, "best_hn_predictor.pth")
    loss_chart = os.path.join(model_dir, "loss_history.png")
    upvote_chart = os.path.join(model_dir, "upvote_predictor_training.png")
    
    # Log charts if available
    if os.path.exists(loss_chart):
        wandb.log({"loss_chart": wandb.Image(loss_chart)})
    
    if os.path.exists(upvote_chart):
        wandb.log({"upvote_training": wandb.Image(upvote_chart)})
    
    # Upload model files
    if os.path.exists(word2vec_file):
        artifact = wandb.Artifact("word2vec-model", type="model")
        artifact.add_file(word2vec_file)
        wandb.log_artifact(artifact)
        print(f"Uploaded Word2Vec model: {word2vec_file}")
    
    if os.path.exists(predictor_file):
        artifact = wandb.Artifact("hn-predictor", type="model")
        artifact.add_file(predictor_file)
        wandb.log_artifact(artifact)
        print(f"Uploaded HN predictor model: {predictor_file}")
        
    print("Upload to W&B complete!")

if __name__ == "__main__":
    # Get API key and model directory from command line arguments
    if len(sys.argv) < 3:
        print("Usage: python upload_to_wandb.py YOUR_API_KEY MODEL_DIRECTORY")
        sys.exit(1)
    
    api_key = sys.argv[1]
    model_dir = sys.argv[2]
    
    # Login to W&B
    wandb.login(key=api_key)
    
    # Upload the model
    upload_model_to_wandb(model_dir)