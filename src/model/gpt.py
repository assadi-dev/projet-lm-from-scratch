"""
gpt.py - Modèle GPT complet

Ce fichier assemble tous les composants pour créer un LLM fonctionnel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embeddings import TransformerEmbedding
from attention import create_causal_mask
from layers import TransformerBlock


class GPT(nn.Module):
    """
    Modèle GPT (Generative Pre-trained Transformer).
    
    Architecture:
        1. Embedding (tokens + positions)
        2. N blocs Transformer
        3. LayerNorm finale
        4. Tête de sortie (prédit le prochain token)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: Taille du vocabulaire (ex: 32000)
            d_model: Dimension du modèle (ex: 256)
            n_heads: Nombre de têtes d'attention (ex: 8)
            n_layers: Nombre de blocs Transformer (ex: 6)
            d_ff: Dimension du feed-forward (ex: 1024)
            max_seq_len: Longueur maximale des séquences (ex: 512)
            dropout: Taux de dropout
        """
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # 1. Embedding (tokens + positions)
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len, dropout)
        
        # 2. Blocs Transformer empilés
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 3. LayerNorm finale
        self.ln_f = nn.LayerNorm(d_model)
        
        # 4. Tête de sortie (projette vers le vocabulaire)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Poids partagés entre embedding et lm_head (weight tying)
        # Cela réduit le nombre de paramètres et améliore les performances
        self.embedding.token_embedding.embedding.weight = self.lm_head.weight
        
        # Initialisation des poids
        self.apply(self._init_weights)
        
        # Afficher le nombre de paramètres
        print(f"Modèle GPT créé avec {self.count_parameters():,} paramètres")
    
    def _init_weights(self, module):
        """Initialisation des poids (comme dans GPT-2)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def count_parameters(self):
        """Compte le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, idx, targets=None):
        """
        Forward pass du modèle.
        
        Args:
            idx: Tensor de shape (batch_size, seq_len) contenant les token IDs
            targets: Tensor optionnel de shape (batch_size, seq_len) pour calculer la loss
        
        Returns:
            logits: Tensor de shape (batch_size, seq_len, vocab_size)
            loss: Scalaire si targets est fourni, None sinon
        """
        batch_size, seq_len = idx.size()
        device = idx.device
        
        # Vérifier la longueur
        assert seq_len <= self.max_seq_len, f"Séquence trop longue: {seq_len} > {self.max_seq_len}"
        
        # 1. Embedding
        x = self.embedding(idx)  # (batch, seq_len, d_model)
        
        # 2. Créer le masque causal
        mask = create_causal_mask(seq_len, device)
        
        # 3. Passer par les blocs Transformer
        for block in self.blocks:
            x = block(x, mask)
        
        # 4. LayerNorm finale
        x = self.ln_f(x)
        
        # 5. Projection vers le vocabulaire
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # 6. Calcul de la loss (si targets fournis)
        loss = None
        if targets is not None:
            # Reshape pour cross_entropy: (batch * seq_len, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Génère du texte de manière autoregressive.
        
        Args:
            idx: Tensor de shape (batch_size, seq_len) - le contexte initial
            max_new_tokens: Nombre de tokens à générer
            temperature: Contrôle l'aléatoire (1.0 = normal, <1 = conservateur, >1 = créatif)
            top_k: Si défini, garde seulement les k tokens les plus probables
        
        Returns:
            Tensor de shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Tronquer si la séquence dépasse max_seq_len
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Prendre les logits du dernier token
            logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # Appliquer la température
            logits = logits / temperature
            
            # Top-k sampling (optionnel)
            if top_k is not None:
                # Garder seulement les top_k valeurs
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_value = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_value, float('-inf'), logits)
            
            # Convertir en probabilités
            probs = F.softmax(logits, dim=-1)
            
            # Échantillonner le prochain token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Ajouter à la séquence
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx


# =============================================================================
# TEST : Vérifie que tout fonctionne
# =============================================================================

if __name__ == "__main__":
    # Paramètres du modèle (petit modèle pour tester)
    vocab_size = 32000
    d_model = 256
    n_heads = 8
    n_layers = 6
    d_ff = 1024
    max_seq_len = 512
    
    # Créer le modèle
    print("=" * 50)
    print("Création du modèle GPT")
    print("=" * 50)
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )
    
    # Test forward pass
    print("\n--- Test Forward Pass ---")
    batch_size = 2
    seq_len = 10
    
    # Simuler des token IDs
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {x.shape}")
    
    logits, loss = model(x, targets)
    
    print(f"Logits shape: {logits.shape}")  # (2, 10, 32000)
    print(f"Loss: {loss.item():.4f}")
    
    # La loss initiale devrait être proche de -ln(1/vocab_size) = ln(32000) ≈ 10.4
    expected_loss = math.log(vocab_size)
    print(f"Loss attendue (random): ~{expected_loss:.2f}")
    
    # Test génération
    print("\n--- Test Génération ---")
    prompt = torch.randint(0, vocab_size, (1, 5))  # 5 tokens de contexte
    print(f"Prompt shape: {prompt.shape}")
    
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")  # (1, 25)
    print(f"Tokens générés: {generated[0, 5:].tolist()}")  # Les 20 nouveaux tokens
    
    # Résumé
    print("\n" + "=" * 50)
    print("RÉSUMÉ DU MODÈLE")
    print("=" * 50)
    print(f"Vocab size: {vocab_size:,}")
    print(f"d_model: {d_model}")
    print(f"n_heads: {n_heads}")
    print(f"n_layers: {n_layers}")
    print(f"d_ff: {d_ff}")
    print(f"max_seq_len: {max_seq_len}")
    print(f"Total paramètres: {model.count_parameters():,}")