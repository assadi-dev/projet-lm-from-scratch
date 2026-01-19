"""
tokenizer.py - Wrapper simple autour de tiktoken

On utilise le tokenizer GPT-2 de OpenAI (50257 tokens).
C'est plus simple que d'entraîner le nôtre pour commencer.
"""

import tiktoken


class Tokenizer:
    """
    Tokenizer basé sur tiktoken (GPT-2).
    
    Convertit du texte en IDs et vice-versa.
    """
    
    def __init__(self):
        # Charger le tokenizer GPT-2
        self.encoder = tiktoken.get_encoding("gpt2")
        
        # Tokens spéciaux
        self.eos_token_id = self.encoder.eot_token  # End of text = 50256
        self.vocab_size = self.encoder.n_vocab      # 50257
    
    def encode(self, text: str) -> list:
        """
        Convertit du texte en liste d'IDs.
        
        Args:
            text: Le texte à encoder
        
        Returns:
            Liste d'entiers (token IDs)
        """
        return self.encoder.encode(text, allowed_special={'<|endoftext|>'})
    
    def decode(self, ids: list) -> str:
        """
        Convertit une liste d'IDs en texte.
        
        Args:
            ids: Liste d'entiers (token IDs)
        
        Returns:
            Le texte décodé
        """
        return self.encoder.decode(ids)
    
    def __len__(self):
        return self.vocab_size


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    tokenizer = Tokenizer()
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    
    # Test encode/decode
    text = "Bonjour, je suis un modèle de langage!"
    print(f"\nTexte original: {text}")
    
    ids = tokenizer.encode(text)
    print(f"Token IDs: {ids}")
    print(f"Nombre de tokens: {len(ids)}")
    
    decoded = tokenizer.decode(ids)
    print(f"Texte décodé: {decoded}")
    
    # Vérifier que c'est identique
    assert text == decoded, "Erreur: le texte décodé ne correspond pas!"
    print("\n✅ Encode/decode fonctionne correctement!")
    
    # Voir les tokens individuels
    print("\nTokens individuels:")
    for id in ids:
        token = tokenizer.decode([id])
        print(f"  {id:5d} -> '{token}'")