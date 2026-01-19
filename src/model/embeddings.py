
import torch
import torch.nn as nn
import math

"""
embeddings.py - Token et Position Embeddings pour notre LLM

L'embedding convertit les IDs de tokens en vecteurs denses.
"""



class TokenEmbedding(nn.Module):
    """
    Convertit les IDs de tokens en vecteurs.
    
    C'est simplement une table de lookup :
    - Chaque token ID correspond à une ligne dans la table
    - Chaque ligne contient d_model nombres (le vecteur)
    """
    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: Nombre total de tokens dans le vocabulaire (ex: 32000)
            d_model: Dimension des vecteurs (ex: 256)
        """
        super().__init__()

        # La table d'embedding : matrice de taille (vocab_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self,x):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len) contenant les IDs
        Returns:
            Embeddings des tokens (tensor de floats)
        """
        # On multiplie par sqrt(d_model) pour stabiliser l'entraînement
        # (astuce du papier "Attention is All You Need")
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """
   Ajoute l'information de position aux tokens.
    
    Sans ça, le modèle ne saurait pas que "chat mange souris" 
    est différent de "souris mange chat".
    
    On utilise des embeddings appris (comme GPT) plutôt que 
    sinusoïdaux (comme le Transformer original).
    """
    def __init__(self, max_seq_len: int, d_model: int):
        """
        Args:
            max_seq_len: Longueur maximale des séquences (ex: 512)
            d_model: Dimension des vecteurs (ex: 256)
        """
        super().__init__()

        # Une embedding pour chaque position possible
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self,seq_len:int,device):
        """
        Args:
            seq_len: Longueur de la séquence actuelle
            device: CPU ou CUDA
        
        Returns:
            Tensor de shape (seq_len, d_model)
        """

        # Créer les positions [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device)

        # Obtenir les embeddings des positions
        return self.embedding(positions)

class TransformerEmbedding(nn.Module):
    """
    Combine Token Embedding + Position Embedding.
    
    C'est la première couche de notre LLM.
    """
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        """
        Args:
            x:Tensor de shape (batch_size,seq_len) contenant les IDs
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """

        seq_len = x.size(1)
        
        # Token embeddings: (batch, seq_len, d_model)
        tok_emb = self.token_embedding(x)
        
        # Position embeddings: (seq_len, d_model)
        pos_emb = self.position_embedding(seq_len, x.device)
        
        # Additionner et appliquer dropout
        # pos_emb est broadcasté sur la dimension batch
        return self.dropout(tok_emb + pos_emb)




# =============================================================================
# TEST : Vérifie que tout fonctionne
# =============================================================================

if __name__ == "__main__":
    # Paramètres
    vocab_size = 32000   # Taille du vocabulaire
    d_model = 256        # Dimension des embeddings
    max_seq_len = 512    # Longueur max des séquences
    batch_size = 2       # Nombre de séquences en parallèle
    seq_len = 10         # Longueur de notre séquence test
    
    # Créer le module d'embedding
    embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len)
    
    # Créer des données de test (IDs aléatoires simulant des tokens)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape: {x.shape}")           # (2, 10)
    print(f"Input (token IDs):\n{x}")
    
    # Forward pass
    output = embedding(x)
    print(f"\nOutput shape: {output.shape}")   # (2, 10, 256)
    print(f"Output (premiers vecteurs):\n{output[0, 0, :10]}")  # 10 premiers nombres
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in embedding.parameters())
    print(f"\nNombre de paramètres: {total_params:,}")
    # vocab_size * d_model + max_seq_len * d_model
    # = 32000 * 256 + 512 * 256 = 8,323,072