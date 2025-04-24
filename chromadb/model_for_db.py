import torch
import torch.nn as nn
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class CBOW(torch.nn.Module):
  def __init__(self, voc, emb):
    super().__init__()
    self.embeddings = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.linear = torch.nn.Linear(in_features=emb, out_features=voc)

  def forward(self, inpt):
    emb = self.embeddings(inpt)
    emb = emb.mean(dim=1)
    out = self.linear(emb)
    return out


class QryTower(nn.Module):
    def __init__(self, embedding_dim=100, hidden_dim=64):
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x, lengths):
        # 1. Sort by descending length
        lengths, sort_idx = lengths.sort(descending=True)
        x = x[sort_idx]

        # 2. Pack
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # 3. GRU
        _, hidden = self.rnn(packed)  # hidden: [1, batch_size, hidden_dim]

        # 4. Unsort
        _, unsort_idx = sort_idx.sort()
        hidden = hidden.squeeze(0)[unsort_idx]  # [batch_size, hidden_dim]

        return hidden

class DocTower(nn.Module):
    def __init__(self, embedding_dim=100, hidden_dim=64):
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x, lengths):
        # 1. Sort by descending length
        lengths, sort_idx = lengths.sort(descending=True)
        x = x[sort_idx]

        # 2. Pack
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # 3. GRU
        _, hidden = self.rnn(packed)  # hidden: [1, batch_size, hidden_dim]

        # 4. Unsort
        _, unsort_idx = sort_idx.sort()
        hidden = hidden.squeeze(0)[unsort_idx]  # [batch_size, hidden_dim]

        return hidden

