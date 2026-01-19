"""
layers.py - Feed-Forward Network et Bloc Transformer

Ce fichier contient :
- FeedForward : réseau à 2 couches qui traite chaque token
- TransformerBlock : combine Attention + FeedForward
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attention import MultiHeadAttention, create_causal_mask


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN).
    
    C'est un simple réseau à 2 couches :
    - Linear: d_model -> d_ff (expansion)
    - GELU activation
    - Linear: d_ff -> d_model (projection)
    
    Généralement d_ff = 4 * d_model
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension du modèle (ex: 256)
            d_ff: Dimension cachée du FFN (ex: 1024)
            dropout: Taux de dropout
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Un bloc Transformer complet.
    
    Architecture (Pre-LayerNorm, comme GPT-2/3):
        x -> LayerNorm -> Attention -> + (résiduel)
          -> LayerNorm -> FeedForward -> + (résiduel) -> output
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension du modèle (ex: 256)
            n_heads: Nombre de têtes d'attention (ex: 8)
            d_ff: Dimension du feed-forward (ex: 1024)
            dropout: Taux de dropout
        """
        super().__init__()
        
        # Couche d'attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, d_model)
            mask: Masque causal optionnel
        
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        # Sous-couche 1 : Attention avec connexion résiduelle
        # x + Attention(LayerNorm(x))
        x = x + self.attention(self.norm1(x), mask)
        
        # Sous-couche 2 : Feed-Forward avec connexion résiduelle
        # x + FeedForward(LayerNorm(x))
        x = x + self.feed_forward(self.norm2(x))
        
        return x


# =============================================================================
# TEST : Vérifie que tout fonctionne
# =============================================================================

if __name__ == "__main__":
    # Paramètres
    d_model = 256
    n_heads = 8
    d_ff = 1024  # 4 * d_model
    batch_size = 2
    seq_len = 10
    
    # Créer un bloc Transformer
    block = TransformerBlock(d_model, n_heads, d_ff)
    
    # Créer des données de test
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")  # (2, 10, 256)
    
    # Créer le masque causal
    mask = create_causal_mask(seq_len, x.device)
    
    # Forward pass
    output = block(x, mask)
    print(f"Output shape: {output.shape}")  # (2, 10, 256)
    
    # Vérifier que la connexion résiduelle fonctionne
    # L'output ne doit pas être trop différent de l'input au début
    diff = (output - x).abs().mean()
    print(f"Différence moyenne input/output: {diff:.4f}")
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in block.parameters())
    print(f"\nNombre de paramètres du bloc: {total_params:,}")
    
    # Tester plusieurs blocs empilés (comme dans un vrai LLM)
    print("\n--- Test avec 6 blocs empilés ---")
    blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(6)])
    
    y = x
    for i, blk in enumerate(blocks):
        y = blk(y, mask)
    
    print(f"Output après 6 blocs: {y.shape}")
    
    total_params_6 = sum(p.numel() for p in blocks.parameters())
    print(f"Paramètres pour 6 blocs: {total_params_6:,}")