"""
generate.py - Script de génération de texte

Charge un modèle et génère du texte à partir d'un prompt.
"""

import torch
import sys
import os

# Ajouter les chemins
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "model"))
sys.path.append(os.path.join(src_path, "data"))

from gpt import GPT
from tokenizer import Tokenizer


class TextGenerator:
    """
    Classe pour générer du texte avec un modèle GPT.
    """
    
    def __init__(self, model: GPT, tokenizer: Tokenizer, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()  # Mode évaluation (désactive dropout)
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> str:
        """
        Génère du texte à partir d'un prompt.
        
        Args:
            prompt: Le texte de départ
            max_new_tokens: Nombre de tokens à générer
            temperature: Créativité (0.1=conservateur, 1.0=créatif)
            top_k: Garde les k meilleurs tokens
        
        Returns:
            Le texte généré (prompt + suite)
        """
        # Encoder le prompt
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        
        # Générer
        with torch.no_grad():
            output_ids = self.model.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        # Décoder
        output_text = self.tokenizer.decode(output_ids[0].tolist())
        
        return output_text
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50
    ):
        """
        Génère du texte token par token (streaming).
        
        Yields:
            Chaque nouveau token généré
        """
        # Encoder le prompt
        prompt_ids = self.tokenizer.encode(prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        
        # Afficher le prompt
        yield prompt
        
        # Générer token par token
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Tronquer si nécessaire
                idx_cond = idx if idx.size(1) <= self.model.max_seq_len else idx[:, -self.model.max_seq_len:]
                
                # Forward
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Top-k
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    min_value = values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_value, float('-inf'), logits)
                
                # Échantillonner
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Ajouter à la séquence
                idx = torch.cat([idx, next_token], dim=1)
                
                # Décoder et yield le nouveau token
                new_token_text = self.tokenizer.decode([next_token.item()])
                yield new_token_text


# =============================================================================
# TEST : Génération avec un modèle non entraîné (random)
# =============================================================================

if __name__ == "__main__":
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Créer tokenizer
    tokenizer = Tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Créer un petit modèle (non entraîné)
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=128
    )
    
    # Créer le générateur
    generator = TextGenerator(model, tokenizer, device)
    
    # Test 1: Génération simple
    print("\n" + "=" * 60)
    print("TEST 1: Génération simple (modèle non entraîné = charabia)")
    print("=" * 60)
    
    prompt = "Le chat"
    print(f"\nPrompt: '{prompt}'")
    print(f"\nGénération:")
    
    output = generator.generate(
        prompt,
        max_new_tokens=30,
        temperature=0.8,
        top_k=50
    )
    print(output)
    
    # Test 2: Génération streaming
    print("\n" + "=" * 60)
    print("TEST 2: Génération streaming (token par token)")
    print("=" * 60)
    
    prompt = "Il était"
    print(f"\nPrompt: '{prompt}'")
    print(f"\nGénération (streaming):")
    
    for token in generator.generate_stream(prompt, max_new_tokens=20, temperature=0.9):
        print(token, end="", flush=True)
    
    print("\n")
    
    # Test 3: Différentes températures
    print("=" * 60)
    print("TEST 3: Effet de la température")
    print("=" * 60)
    
    prompt = "Bonjour"
    
    for temp in [0.3, 0.7, 1.0, 1.5]:
        print(f"\nTempérature = {temp}:")
        output = generator.generate(prompt, max_new_tokens=20, temperature=temp, top_k=50)
        print(f"  {output}")
    
    print("\n✅ Générateur fonctionne!")
    print("\nNote: Le texte est du charabia car le modèle n'est pas entraîné.")
    print("Après entraînement, il générera du texte cohérent.")