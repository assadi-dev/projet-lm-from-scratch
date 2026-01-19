# Architecture Transformer - Guide détaillé

## Table des matières

1. Vue d'ensemble de l'architecture
2. Embeddings et encodage positionnel
3. Multi-Head Self-Attention
4. Feed-Forward Network
5. Layer Normalization et connexions résiduelles
6. Assembler le modèle complet

---

## 1. Vue d'ensemble de l'architecture

Un LLM decoder-only suit ce flux:

```
Input IDs → Embedding → [Bloc Transformer × N] → Linear → Logits
```

Chaque bloc Transformer contient:
- Multi-Head Self-Attention (masquée)
- Feed-Forward Network
- Layer Normalization (Pre-LN recommandé)
- Connexions résiduelles

---

## 2. Embeddings et encodage positionnel

### Token Embeddings

```python
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
```

### Encodage positionnel sinusoïdal

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Créer la matrice de positions
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

### Alternative: Rotary Position Embedding (RoPE)

Plus moderne, utilisé par LLaMA et autres:

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

---

## 3. Multi-Head Self-Attention

### Attention de base

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, n_heads, seq_len, d_k)
    mask: (batch, 1, seq_len, seq_len) ou (1, 1, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

### Multi-Head Attention complet

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Projections linéaires
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape pour multi-head: (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatener les têtes
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Projection de sortie
        output = self.W_o(attn_output)
        return self.dropout(output)
```

### Masque causal

Pour la génération autoregressive, on empêche le modèle de voir les tokens futurs:

```python
def create_causal_mask(seq_len, device):
    """Crée un masque triangulaire inférieur"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
```

---

## 4. Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # ou nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)
```

### Variante SwiGLU (plus moderne)

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))
```

---

## 5. Bloc Transformer complet

### Pre-LayerNorm (recommandé)

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Pre-LN: normaliser AVANT l'opération
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x
```

---

## 6. Modèle complet

```python
class GPT(nn.Module):
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
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Blocs Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Couche finale
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (optionnel mais recommandé)
        self.token_embedding.weight = self.head.weight
        
        # Initialisation
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()
        assert seq_len <= self.max_seq_len, f"Séquence trop longue: {seq_len} > {self.max_seq_len}"
        
        # Embeddings
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Masque causal
        mask = torch.triu(torch.ones(seq_len, seq_len, device=idx.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Blocs Transformer
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Calcul de la loss si targets fournis
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Pour le padding
            )
        
        return logits, loss
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

## Formules de calcul des paramètres

Pour un modèle avec:
- `V` = vocab_size
- `D` = d_model
- `H` = n_heads
- `L` = n_layers
- `F` = d_ff

Nombre de paramètres ≈ `L × (4D² + 2DF) + V×D`

Exemple avec les hyperparamètres recommandés:
- V=32000, D=256, H=8, L=6, F=1024
- ≈ 6 × (4×256² + 2×256×1024) + 32000×256
- ≈ 6 × (262144 + 524288) + 8192000
- ≈ 4.7M + 8.2M ≈ **13M paramètres**
