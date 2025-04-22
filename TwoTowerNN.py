import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class QryTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input dimension: 128
        # Hidden layers: 128 -> 64
        # Output dimension: 64
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DocTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input dimension: 128
        # Hidden layers: 128 -> 64
        # Output dimension: 64
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 64)
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
