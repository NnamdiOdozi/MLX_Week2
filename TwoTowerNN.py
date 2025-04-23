import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader,  SubsetRandomSampler
from sklearn.model_selection import KFold, train_test_split
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()
import wandb
import datetime

class QryTower(torch.nn.Module):
    def __init__(self, input_dim=128, hidden_dims=[128, 64], output_dim=64):
        super().__init__()
        # Input dimension: 128
        # Hidden layers: 128 -> 64
        # Output dimension: 64
        self.fc1 = torch.nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = torch.nn.Linear(hidden_dims[1], output_dim)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DocTower(torch.nn.Module):
    def __init__(self, input_dim=128, hidden_dims=[128, 64], output_dim=64):
        super().__init__()
        # Input dimension: 128
        # Hidden layers: 128 -> 64
        # Output dimension: 64
        self.fc1 = torch.nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = torch.nn.Linear(hidden_dims[1], output_dim)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TripletEmbeddingDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Convert list embeddings to tensors
        query_embedding = torch.tensor(row['avg_query_embedding'], dtype=torch.float32)
        pos_embedding = torch.tensor(row['avg_pos_embedding'], dtype=torch.float32)
        neg_embedding = torch.tensor(row['avg_neg_embedding'], dtype=torch.float32)
        
        return {
            'query': query_embedding,
            'positive': pos_embedding,
            'negative': neg_embedding
        }


# 3. Function to train a model with specific hyperparameters
def train_model(train_loader, val_loader, output_dim, lr=1e-3, epochs=10, checkpoint_dir="checkpoints", log_wandb=True):
    
      
    # Create models
    qryTower = QryTower(output_dim=output_dim)
    docTower = DocTower(output_dim=output_dim)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW([
        {'params': qryTower.parameters()},
        {'params': docTower.parameters()}
    ], lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        qryTower.train()
        docTower.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            query_emb = batch['query']
            pos_emb = batch['positive']
            neg_emb = batch['negative']
            
            # Forward pass
            query_encoded = qryTower(query_emb)
            pos_encoded = docTower(pos_emb)
            neg_encoded = docTower(neg_emb)
            
            # Calculate similarities
            pos_sim = torch.nn.functional.cosine_similarity(query_encoded, pos_encoded)
            neg_sim = torch.nn.functional.cosine_similarity(query_encoded, neg_encoded)
            
            # Triplet loss
            margin = 0.2
            loss = torch.clamp(margin - pos_sim + neg_sim, min=0).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(query_emb)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        qryTower.eval()
        docTower.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                query_emb = batch['query']
                pos_emb = batch['positive']
                neg_emb = batch['negative']
                
                # Forward pass
                query_encoded = qryTower(query_emb)
                pos_encoded = docTower(pos_emb)
                neg_encoded = docTower(neg_emb)
                
                # Calculate similarities
                pos_sim = torch.nn.functional.cosine_similarity(query_encoded, pos_encoded)
                neg_sim = torch.nn.functional.cosine_similarity(query_encoded, neg_encoded)
                
                # Triplet loss
                margin = 0.2
                loss = torch.clamp(margin - pos_sim + neg_sim, min=0).mean()
                
                val_loss += loss.item() * len(query_emb)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint for every epoch (as a separate checkpoint)
        checkpoint = {
            'epoch': epoch,
            'qryTower_state_dict': qryTower.state_dict(),
            'docTower_state_dict': docTower.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'hyperparams': {
                'output_dim': output_dim,
                'lr': lr
            }
        }       
        
        if log_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Save the latest checkpoint
        #torch.save(checkpoint, f"{checkpoint_dir}/latest_checkpoint.pt")
        
        # Save the best model (only if this is the best we've seen)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, f"{checkpoint_dir}/best_model_dim{output_dim}_lr{lr}.pt")

            # Optionally log model to W&B
            if log_wandb:
                model_artifact = wandb.Artifact(
                    name=f"model_dim{output_dim}_lr{lr}_epoch{epoch+1}", 
                    type="model",
                    description=f"Model with val_loss: {best_val_loss:.4f}"
                )
                model_artifact.add_file(f"{checkpoint_dir}/best_model_dim{output_dim}_lr{lr}.pt")
                wandb.log_artifact(model_artifact)

            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    


    return best_val_loss, qryTower, docTower

# 4. Main hyperparameter tuning with cross-validation
def run_hyperparameter_tuning(df, output_dims=[32, 64, 128], batch_sizes=[128, 256, 512], n_folds=5, epochs=10):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Initialize W&B for the entire tuning process
    wandb.init(
        project="twin-tower-model",
        name=f"hyperparameter-tuning {timestamp}",
        config={
            "output_dims": output_dims,
            "batch_sizes": batch_sizes,
            "n_folds": n_folds,
            "epochs": epochs
        }
    )

    # First, create a train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create the full dataset
    full_dataset = TripletEmbeddingDataset(train_df)
    
    # Set up k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Results tracking
    results = []
    
    # Loop through hyperparameter combinations
    for output_dim in output_dims:
        for batch_size in batch_sizes:
            print(f"\n\n{'-'*50}")
            print(f"Training with output_dim={output_dim}, batch_size={batch_size}")
            print(f"{'-'*50}")
            
            # Cross validation
            fold_losses = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset)))):
                print(f"\nFold {fold+1}/{n_folds}")
                
                # Create data samplers
                train_sampler = SubsetRandomSampler(train_idx)
                val_sampler = SubsetRandomSampler(val_idx)
                
                # Create data loaders
                train_loader = DataLoader(
                    full_dataset, 
                    batch_size=batch_size, 
                    sampler=train_sampler,
                    num_workers=8,
                    pin_memory=True
                )
                
                val_loader = DataLoader(
                    full_dataset, 
                    batch_size=batch_size, 
                    sampler=val_sampler,
                    num_workers=2,
                    pin_memory=True
                )
                
                # Train the model and get validation loss
                checkpoint_dir = f"checkpoints/dim{output_dim}_batch{batch_size}_fold{fold+1}_{timestamp}"
                val_loss, _, _ = train_model(
                    train_loader, 
                    val_loader, 
                    output_dim=output_dim,
                    epochs=epochs,
                    checkpoint_dir=checkpoint_dir,
                    log_wandb=True  # Disable W&B logging for fold training
                )
                
                fold_losses.append(val_loss)
            
            # Calculate average loss across folds
            avg_cv_loss = np.mean(fold_losses)
            print(f"\nAverage CV loss for output_dim={output_dim}, batch_size={batch_size}: {avg_cv_loss:.4f}")

            # Log the cross-validation results
            wandb.log({
                "output_dim": output_dim,
                "batch_size": batch_size,
                "avg_cv_loss": avg_cv_loss,
                "fold_losses": fold_losses
            })
            
            # Record result
            results.append({
                'output_dim': output_dim,
                'batch_size': batch_size,
                'avg_cv_loss': avg_cv_loss,
                'fold_losses': fold_losses
            })
    
    # Find best hyperparameters
    best_result = min(results, key=lambda x: x['avg_cv_loss'])
    print(f"\n\nBest hyperparameters:")
    print(f"Output dimension: {best_result['output_dim']}")
    print(f"Batch size: {best_result['batch_size']}")
    print(f"Average CV loss: {best_result['avg_cv_loss']:.4f}")
    
    # Train final model on all training data with best hyperparameters
    print("\n\nTraining final model with best hyperparameters...")
    full_train_dataset = TripletEmbeddingDataset(train_df)
    test_dataset = TripletEmbeddingDataset(test_df)
    
    train_loader = DataLoader(
        full_train_dataset, 
        batch_size=best_result['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=best_result['batch_size'], 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Train final model
    _, final_qry_tower, final_doc_tower = train_model(
        train_loader, 
        test_loader, 
        output_dim=best_result['output_dim'],
        epochs=epochs,
        checkpoint_dir="checkpoints/final_model"
    )
    
    # Save final model
    final_model = {
        'qryTower_state_dict': final_qry_tower.state_dict(),
        'docTower_state_dict': final_doc_tower.state_dict(),
        'hyperparams': {
            'output_dim': best_result['output_dim'],
            'batch_size': best_result['batch_size']
        }
    }
    torch.save(final_model, "checkpoints/final_model/final_model.pt")

    # Before returning, finish the W&B run
    wandb.finish()
    
    return best_result, final_qry_tower, final_doc_tower
