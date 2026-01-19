"""
transformer.py - Architecture GPT complète

Ce fichier contient l'implémentation complète d'un modèle GPT-like
pour la génération de texte.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """Configuration du modèle GPT."""
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False  # True pour inclure les biais dans les Linear


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention avec masque causal."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.d_model = config.d_model
        
        # Projections Q, K, V combinées pour efficacité
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Masque causal
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, seq_len, d_model
        
        # Calculer Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape pour multi-head: (B, T, n_heads, d_k) -> (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = att.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Appliquer l'attention
        y = att @ v  # (B, n_heads, T, d_k)
        
        # Concatener les têtes
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Projection de sortie
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """Feed-Forward Network avec GELU."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Bloc Transformer avec Pre-LayerNorm."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Modèle GPT complet."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)  # token
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)  # position
        self.drop = nn.Dropout(config.dropout)
        
        # Blocs Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Layer norm finale
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Tête de sortie (LM head)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        # Initialisation
        self.apply(self._init_weights)
        
        # Scaling spécial pour les projections résiduelles
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))
        
        print(f"Nombre de paramètres: {self.get_num_params():,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Compte les paramètres."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx: (B, T) tensor d'indices de tokens
            targets: (B, T) tensor de targets pour la loss
        
        Returns:
            logits: (B, T, vocab_size)
            loss: scalaire si targets fourni
        """
        device = idx.device
        B, T = idx.size()
        
        assert T <= self.config.max_seq_len, f"Séquence trop longue ({T} > {self.config.max_seq_len})"
        
        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.wte(idx)  # (B, T, d_model)
        pos_emb = self.wpe(pos)  # (T, d_model)
        x = self.drop(tok_emb + pos_emb)
        
        # Blocs Transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # Calcul des logits
        if targets is not None:
            # Entraînement: calculer loss sur tous les tokens
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inférence: calculer seulement le dernier token
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Génère des tokens autoregressivement.
        
        Args:
            idx: (B, T) contexte initial
            max_new_tokens: nombre de tokens à générer
            temperature: contrôle l'aléatoire
            top_k: garder les k meilleurs tokens
            top_p: nucleus sampling
        """
        for _ in range(max_new_tokens):
            # Tronquer si nécessaire
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# Test rapide
if __name__ == "__main__":
    config = GPTConfig(
        vocab_size=32000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )
    
    model = GPT(config)
    
    # Test forward
    x = torch.randint(0, config.vocab_size, (2, 64))  # batch=2, seq=64
    y = torch.randint(0, config.vocab_size, (2, 64))
    
    logits, loss = model(x, y)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test génération
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
