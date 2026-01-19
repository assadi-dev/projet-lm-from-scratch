"""
attention.py - Multi-Head Self-Attention

Le mécanisme d'attention est le coeur du Transformer.
Il permet à chaque token de "regarder" les autres tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):

    """
    Multi-Head Self-Attention avec masque causal.
    
    Le masque causal empêche le modèle de "tricher" en regardant
    les tokens futurs pendant la génération.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension du modèle (ex: 256)
            n_heads: Nombre de têtes d'attention (ex: 8)
            dropout: Taux de dropout (ex: 0.1)
        """
        super().__init__()

        # Vérifier que d_model est divisible par n_heads
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"

                
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension par tête (ex: 256/8 = 32)
        
        # Projections linéaires pour Q, K, V (combinées pour efficacité)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Projection de sortie
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, d_model)
            mask: Masque optionnel (batch_size, 1, seq_len, seq_len)
        
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()

                
        # 1. Projections linéaires
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. Reshape pour multi-head
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Calcul de l'attention
        # scores = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: (batch, n_heads, seq_len, seq_len)
        
        # 4. Appliquer le masque (si fourni)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax pour obtenir les poids d'attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 6. Appliquer l'attention aux valeurs
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_k)
        # -> (batch, n_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, V)
        
        # 7. Concatener les têtes
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # 8. Projection de sortie
        output = self.W_o(attn_output)
        
        return output


def create_causal_mask(seq_len: int, device):
    """
    Crée un masque causal (triangulaire inférieur).
    
    Le masque empêche chaque position de voir les positions futures.
    
    Exemple pour seq_len=4:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    
    1 = peut voir, 0 = ne peut pas voir
    """
    # torch.tril = triangulaire inférieur
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    # Ajouter dimensions pour batch et heads: (1, 1, seq_len, seq_len)
    return mask.unsqueeze(0).unsqueeze(0)



      