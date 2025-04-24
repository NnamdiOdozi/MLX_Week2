import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import numpy as np
import pandas as pd


class QryTower(torch.nn.Module):
    def __init__(self, input_dim=100, hidden_dims=[128, 64], output_dim=100):
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
    def __init__(self, input_dim=100, hidden_dims=[128, 64], output_dim=100):
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


class BidirectionalGRU(nn.Module):
    """Bidirectional GRU for encoding sequences"""
    def __init__(self, embedding_dim, hidden_dim, num_layers=1, dropout=0.0): #layers could be one or two
        super(BidirectionalGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, embeddings, lengths):
        """
        Args:
            embeddings: Embedding tensors [batch_size, seq_len, embedding_dim]
            lengths: Length of each sequence in the batch
            
        Returns:
            encoding: GRU encoding [batch_size, hidden_dim*2]
        """
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, 
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Forward through GRU
        _, hidden = self.gru(packed)
        
        # Get the last layer's forward and backward states
        hidden = hidden.transpose(0, 1)  # [num_layers*2, batch, hidden] -> [batch, num_layers*2, hidden]
        
        # Concatenate the last layer's forward and backward states
        last_forward = hidden[:, -2, :]
        last_backward = hidden[:, -1, :]
        encoding = torch.cat([last_forward, last_backward], dim=1)
        
        return encoding


class GRUTwinTowerModel(nn.Module):
    """Combined model with GRU encoding and twin towers, using pre-computed embeddings"""
    def __init__(
        self,
        embedding_dim=100,
        gru_hidden_dim=128,
        tower_hidden_dims=[128, 64],
        output_dim=100,
        num_layers=1,
        dropout=0.1,
        max_query_length=26,
        max_doc_length=201
    ):
        super(GRUTwinTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        
        # GRU encoders
        self.query_encoder = BidirectionalGRU(
            embedding_dim=embedding_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.doc_encoder = BidirectionalGRU(
            embedding_dim=embedding_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Twin towers (reusing your existing QryTower and DocTower classes)
        self.query_tower = QryTower(
            input_dim=gru_hidden_dim*2,  # *2 because of bidirectional
            hidden_dims=tower_hidden_dims,
            output_dim=output_dim
        )
        
        self.doc_tower = DocTower(
            input_dim=gru_hidden_dim*2,  # *2 because of bidirectional
            hidden_dims=tower_hidden_dims,
            output_dim=output_dim
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query_embeddings,
        query_lengths,
        doc_embeddings,
        doc_lengths
    ):
        """
        Forward pass using pre-computed embeddings
        
        Args:
            query_embeddings: Query embedding sequences [batch_size, seq_len, embedding_dim]
            query_lengths: Length of each query 
            doc_embeddings: Document embedding sequences [batch_size, seq_len, embedding_dim]
            doc_lengths: Length of each document
            
        Returns:
            query_vector: Final query representation [batch_size, output_dim]
            doc_vector: Final document representation [batch_size, output_dim]
        """
        # Encode sequences with GRUs
        query_encoded = self.query_encoder(query_embeddings, query_lengths)
        doc_encoded = self.doc_encoder(doc_embeddings, doc_lengths)
        
        # Apply tower networks
        query_vector = self.query_tower(self.dropout(query_encoded))
        doc_vector = self.doc_tower(self.dropout(doc_encoded))
        
        # Normalize for cosine similarity
        query_vector = F.normalize(query_vector, p=2, dim=1)
        doc_vector = F.normalize(doc_vector, p=2, dim=1)
        
        return query_vector, doc_vector
    
    def prepare_embeddings(self, raw_embeddings, max_length): #I'm not sure i will actually use this fn since the embeddings lists are likely to have been padded before reaching the RNN 
        """
        Convert raw embedding lists from your dataframe to padded tensors
        
        Args:
            raw_embeddings: List of embedding tensors from your dataframe
            max_length: Maximum sequence length to pad to
            
        Returns:
            padded_embeddings: Padded tensor [batch_size, max_length, embedding_dim]
            lengths: Actual sequence lengths [batch_size]
        """
        batch_size = len(raw_embeddings)
        
        # Get sequence lengths
        lengths = torch.tensor([len(emb) for emb in raw_embeddings])
        
        # Create padded tensor
        padded_embeddings = torch.zeros(batch_size, max_length, self.embedding_dim)
        
        # Fill in with actual embeddings
        for i, emb_seq in enumerate(raw_embeddings):
            seq_len = min(len(emb_seq), max_length)
            padded_embeddings[i, :seq_len] = torch.stack(emb_seq[:seq_len])
        
        return padded_embeddings, lengths

class EmbeddingTripletDataset(Dataset):
    """Dataset for pre-computed embedding sequences"""
    def __init__(self, dataframe):
        """
        Args:
            dataframe: DataFrame containing query_emb, pos_emb, neg_emb columns
        """
        self.dataframe = dataframe
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Get embedding lists from dataframe (already tensors)
        query_emb = row['query_emb']
        pos_emb = row['pos_emb']
        neg_emb = row['neg_emb']
        
        return {
            'query_emb': row['query_emb'],
            'query_length': row['query_length'],
            'pos_emb': row['pos_emb'],
            'pos_length': row['pos_length'],
            'neg_emb': row['neg_emb'],
            'neg_length': row['neg_length']
        }


def train_gru_model(train_loader, val_loader, embedding_dim=100, 
                    gru_hidden_dim=128, output_dim=128, 
                    num_layers=1, dropout=0.1,
                    lr=1e-3, epochs=10, margin=0.2,
                    max_query_length=26, max_doc_length=201,
                    checkpoint_dir="checkpoints", log_wandb=True):
    """
    Train GRU-based twin tower model using pre-computed embeddings
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        embedding_dim: Dimension of embedding vectors
        gru_hidden_dim: Hidden dimension of GRU
        output_dim: Output dimension of twin towers
        num_layers: Number of GRU layers
        dropout: Dropout rate
        lr: Learning rate
        epochs: Number of training epochs
        margin: Margin for triplet loss
        max_query_length: Maximum query length for padding
        max_doc_length: Maximum document length for padding
        checkpoint_dir: Directory to save checkpoints
        log_wandb: Whether to log to wandb
    """
    # Create the model
    model = GRUTwinTowerModel(
        embedding_dim=embedding_dim,
        gru_hidden_dim=gru_hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_query_length=max_query_length,
        max_doc_length=max_doc_length
    )
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device
    model = model.to(device)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            # Get batch data (already embedding lists)
            query_embs = batch['query_emb']
            pos_embs = batch['pos_emb']
            neg_embs = batch['neg_emb']
            
            # Prepare embeddings (pad and get lengths) - For now just set these to the inputs
            #query_embeddings, query_lengths = query_embs, query_lengths #model.prepare_embeddings(query_embs, max_query_length)
            #pos_embeddings, pos_lengths = pos_embs, pos_lengths #model.prepare_embeddings(pos_embs, max_doc_length)
            #neg_embeddings, neg_lengths = neg_embs, neg_lengths #model.prepare_embeddings(neg_embs, max_doc_length)
            
            query_embeddings = batch['query_emb']
            query_lengths = batch['query_length']
            pos_embeddings = batch['pos_emb']
            pos_lengths = batch['pos_length']
            neg_embeddings = batch['neg_emb']
            neg_lengths = batch['neg_length']


            # Move data to device
            query_embeddings = query_embeddings.to(device)
            query_lengths = query_lengths.to(device)
            pos_embeddings = pos_embeddings.to(device)
            pos_lengths = pos_lengths.to(device)
            neg_embeddings = neg_embeddings.to(device)
            neg_lengths = neg_lengths.to(device)
            
            # Forward pass for query with positive document
            query_vector_pos, pos_doc_vector = model(
                query_embeddings, query_lengths,
                pos_embeddings, pos_lengths
            )
            
            # Forward pass for query with negative document
            query_vector_neg, neg_doc_vector = model(
                query_embeddings, query_lengths,
                neg_embeddings, neg_lengths
            )
            
            # Calculate similarities
            pos_sim = torch.sum(query_vector_pos * pos_doc_vector, dim=1)
            neg_sim = torch.sum(query_vector_neg * neg_doc_vector, dim=1)
            
            # Triplet loss
            loss = torch.clamp(margin - pos_sim + neg_sim, min=0).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(query_embs)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                # Get batch data
                query_embs = batch['query_emb']
                pos_embs = batch['pos_emb']
                neg_embs = batch['neg_emb']
                
                # Prepare embeddings
                query_embeddings = batch['query_emb']
                query_lengths = batch['query_length']
                pos_embeddings = batch['pos_emb']
                pos_lengths = batch['pos_length']
                neg_embeddings = batch['neg_emb']
                neg_lengths = batch['neg_length']
                # Prepare embeddings (pad and get lengths)                
                # Move data to device
                
                query_embeddings = query_embeddings.to(device)
                query_lengths = query_lengths.to(device)
                pos_embeddings = pos_embeddings.to(device)
                pos_lengths = pos_lengths.to(device)
                neg_embeddings = neg_embeddings.to(device)
                neg_lengths = neg_lengths.to(device)
                
                # Forward pass
                query_vector_pos, pos_doc_vector = model(
                    query_embeddings, query_lengths,
                    pos_embeddings, pos_lengths
                )
                
                query_vector_neg, neg_doc_vector = model(
                    query_embeddings, query_lengths,
                    neg_embeddings, neg_lengths
                )
                
                # Calculate similarities
                pos_sim = torch.sum(query_vector_pos * pos_doc_vector, dim=1)
                neg_sim = torch.sum(query_vector_neg * neg_doc_vector, dim=1)
                
                # Triplet loss
                loss = torch.clamp(margin - pos_sim + neg_sim, min=0).mean()
                
                val_loss += loss.item() * len(query_embs)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'hyperparams': {
                'embedding_dim': embedding_dim,
                'gru_hidden_dim': gru_hidden_dim,
                'output_dim': output_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'lr': lr
            }
        }
        
        # Log to wandb if enabled
        if log_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, f"{checkpoint_dir}/best_gru_model.pt")
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    return best_val_loss, model

def run_hyperparameter_tuning(df, output_dims=[100], batch_sizes=[512], gru_hidden_dims=[128], 
                         num_layers=[1], dropouts=[0.1], learning_rates=[1e-3], 
                         epochs=10, log_wandb=True):
    """
    Run hyperparameter tuning for the GRU Twin Tower model.
    
    Args:
        df: DataFrame with query_emb, pos_emb, neg_emb columns (pre-computed embeddings)
        output_dims: List of output dimensions to try
        batch_sizes: List of batch sizes to try
        gru_hidden_dims: List of GRU hidden dimensions to try
        num_layers: List of number of GRU layers to try
        dropouts: List of dropout rates to try
        learning_rates: List of learning rates to try
        epochs: Number of epochs to train for each combination
        log_wandb: Whether to log to wandb
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Initialize W&B for the entire tuning process
    if log_wandb:
        import wandb
        wandb.init(
            project="gru-twin-tower-model",
            name=f"hyperparameter-tuning-{timestamp}",
            config={
                "output_dims": output_dims,
                "batch_sizes": batch_sizes,
                "gru_hidden_dims": gru_hidden_dims,
                "num_layers": num_layers,
                "dropouts": dropouts,
                "learning_rates": learning_rates,
                "epochs": epochs
            }
        )

    # First, create train/val/test split (60/20/20)
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Data splits: Train={len(train_df)} | Validation={len(val_df)} | Test={len(test_df)}")
    
    # Results tracking
    results = []
    
    # Custom collate function for embedding sequences
    def collate_embedding_batches(batch):
        query_embs = torch.stack([item['query_emb'] for item in batch])
        query_lengths = torch.tensor([item['query_length'] for item in batch])
        
        pos_embs = torch.stack([item['pos_emb'] for item in batch])
        pos_lengths = torch.tensor([item['pos_length'] for item in batch])
        
        neg_embs = torch.stack([item['neg_emb'] for item in batch])
        neg_lengths = torch.tensor([item['neg_length'] for item in batch])
        
        return {
            'query_emb': query_embs,
            'query_length': query_lengths,
            'pos_emb': pos_embs,
            'pos_length': pos_lengths,
            'neg_emb': neg_embs,
            'neg_length': neg_lengths
        }
    
    # Loop through hyperparameter combinations
    for output_dim in output_dims:
        for batch_size in batch_sizes:
            for gru_hidden_dim in gru_hidden_dims:
                for n_layers in num_layers:
                    for dropout in dropouts:
                        for lr in learning_rates:
                            
                            # Initialize a new wandb run for each hyperparameter combination
                            if log_wandb:
                                import wandb
                                run_name = f"dim{output_dim}_batch{batch_size}_hidden{gru_hidden_dim}_layers{n_layers}_{timestamp}"
                                wandb.init(
                                    project="gru-twin-tower-model",
                                    name=run_name,
                                    config={
                                        "output_dim": output_dim,
                                        "batch_size": batch_size,
                                        "gru_hidden_dim": gru_hidden_dim,
                                        "num_layers": n_layers,
                                        "dropout": dropout,
                                        "learning_rate": lr,
                                        "epochs": epochs
                                    },
                                    reinit=True  # Allow multiple wandb runs
                                )
             
                           
                            print(f"\n\n{'-'*80}")
                            print(f"Training with: output_dim={output_dim}, batch_size={batch_size}, " + 
                                  f"gru_hidden_dim={gru_hidden_dim}, num_layers={n_layers}, " + 
                                  f"dropout={dropout}, lr={lr}")
                            print(f"{'-'*80}")
                            
                            # Create datasets
                            train_dataset = EmbeddingTripletDataset(train_df)
                            val_dataset = EmbeddingTripletDataset(val_df)
                            
                            # Create data loaders
                            train_loader = DataLoader(
                                train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True,
                                collate_fn=collate_embedding_batches,
                                num_workers=4,
                                pin_memory=True
                            )
                            
                            val_loader = DataLoader(
                                val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False,
                                collate_fn=collate_embedding_batches,
                                num_workers=2,
                                pin_memory=True
                            )
                            
                            # Train the model
                            checkpoint_dir = f"checkpoints/gru_dim{output_dim}_batch{batch_size}_hidden{gru_hidden_dim}_layers{n_layers}_{timestamp}"
                            val_loss, _ = train_gru_model(
                                train_loader, 
                                val_loader, 
                                embedding_dim=100,  # Assuming GloVe dim is 100, adjust if different
                                gru_hidden_dim=gru_hidden_dim,
                                output_dim=output_dim,
                                num_layers=n_layers,
                                dropout=dropout,
                                lr=lr,
                                epochs=epochs,
                                checkpoint_dir=checkpoint_dir,
                                log_wandb=True  # We'll log at this level instead
                            )
                            
                            # Log the results
                            if log_wandb:
                                wandb.finish()
                            
                            # Record result
                            results.append({
                                'output_dim': output_dim,
                                'batch_size': batch_size,
                                'gru_hidden_dim': gru_hidden_dim,
                                'num_layers': n_layers,
                                'dropout': dropout,
                                'learning_rate': lr,
                                'val_loss': val_loss
                            })
    
    # Find best hyperparameters
    best_result = min(results, key=lambda x: x['val_loss'])
    print(f"\n\nBest hyperparameters:")
    print(f"Output dimension: {best_result['output_dim']}")
    print(f"Batch size: {best_result['batch_size']}")
    print(f"GRU hidden dimension: {best_result['gru_hidden_dim']}")
    print(f"Number of GRU layers: {best_result['num_layers']}")
    print(f"Dropout: {best_result['dropout']}")
    print(f"Learning rate: {best_result['learning_rate']}")
    print(f"Validation Loss: {best_result['val_loss']:.4f}")
    
    # Train final model on combined train+val data with best hyperparameters
    print("\n\nTraining final model with best hyperparameters...")
    
    # Combine train and validation data for final training
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    full_train_dataset = EmbeddingTripletDataset(train_val_df)
    test_dataset = EmbeddingTripletDataset(test_df)
    
    # Create data loaders for final training
    train_loader = DataLoader(
        full_train_dataset, 
        batch_size=best_result['batch_size'], 
        shuffle=True,
        collate_fn=collate_embedding_batches,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=best_result['batch_size'], 
        shuffle=False,
        collate_fn=collate_embedding_batches,
        num_workers=2,
        pin_memory=True
    )
    
    # Train final model
    final_checkpoint_dir = f"checkpoints/final_gru_model_{timestamp}"
    _, final_model = train_gru_model(
        train_loader, 
        test_loader, 
        embedding_dim=100,  # Adjust if your GloVe dim is different
        gru_hidden_dim=best_result['gru_hidden_dim'],
        output_dim=best_result['output_dim'],
        num_layers=best_result['num_layers'],
        dropout=best_result['dropout'],
        lr=best_result['learning_rate'],
        epochs=epochs,
        checkpoint_dir=final_checkpoint_dir,
        log_wandb=False
    )
    
    # Save final model
    final_model_path = f"{final_checkpoint_dir}/final_gru_model_{timestamp}.pt"
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'hyperparams': best_result
    }, final_model_path)
    
    print(f"Final model saved at: {final_model_path}")

    # Log final model to W&B
    if log_wandb:
        final_model_artifact = wandb.Artifact(
            name=f"final_gru_model_{timestamp}", 
            type="model",
            description=f"Final GRU model trained on combined data with best hyperparameters"
        )
        
        final_model_artifact.add_file(final_model_path)
        wandb.log_artifact(final_model_artifact)
        
        # Finish the W&B run
        wandb.finish()
    
    return best_result, final_model