"""
dataset.py - Dataset PyTorch pour l'entraînement du LLM

Charge du texte, le tokenise, et crée des paires (input, target)
pour apprendre à prédire le token suivant.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tokenizer import Tokenizer


class TextDataset(Dataset):
    """
    Dataset pour le language modeling.
    
    Prend du texte tokenisé et crée des fenêtres de taille fixe.
    
    Exemple avec seq_len=4:
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        
        Échantillon 0: input=[1,2,3,4], target=[2,3,4,5]
        Échantillon 1: input=[2,3,4,5], target=[3,4,5,6]
        ...
    """
    
    def __init__(self, text: str, tokenizer: Tokenizer, seq_len: int = 512):
        """
        Args:
            text: Le texte brut à utiliser
            tokenizer: Le tokenizer pour convertir texte -> IDs
            seq_len: Longueur des séquences
        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        # Tokeniser tout le texte
        self.tokens = tokenizer.encode(text)
        
        print(f"Dataset créé: {len(self.tokens):,} tokens")
    
    def __len__(self):
        # Nombre d'échantillons possibles
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        """
        Retourne une paire (input, target).
        
        input:  tokens[idx : idx + seq_len]
        target: tokens[idx + 1 : idx + seq_len + 1]
        """
        # Extraire la fenêtre
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        
        # Séparer input et target
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


def create_dataloader(
    text: str,
    tokenizer: Tokenizer,
    seq_len: int = 512,
    batch_size: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Crée un DataLoader à partir de texte brut.
    """
    dataset = TextDataset(text, tokenizer, seq_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 0 pour Windows
        pin_memory=True
    )
    
    return dataloader


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Créer le tokenizer
    tokenizer = Tokenizer()
    
    # Texte d'exemple (en vrai, on utiliserait un gros fichier)
    sample_text = """
    Il était une fois un petit chat qui vivait dans une grande maison.
    Ce chat aimait dormir toute la journée et jouer la nuit.
    Un jour, le chat décida de partir à l'aventure.
    Il rencontra un chien, un oiseau et une souris.
    Ensemble, ils devinrent les meilleurs amis du monde.
    Le chat était très heureux de ses nouveaux amis.
    """ * 100  # Répéter pour avoir plus de données
    
    print(f"Texte: {len(sample_text)} caractères")
    
    # Créer le dataset
    seq_len = 64  # Séquences courtes pour le test
    dataset = TextDataset(sample_text, tokenizer, seq_len)
    
    print(f"Nombre d'échantillons: {len(dataset)}")
    
    # Tester un échantillon
    x, y = dataset[0]
    print(f"\nÉchantillon 0:")
    print(f"  Input shape: {x.shape}")   # (64,)
    print(f"  Target shape: {y.shape}")  # (64,)
    print(f"  Input (premiers tokens): {x[:10].tolist()}")
    print(f"  Target (premiers tokens): {y[:10].tolist()}")
    
    # Vérifier que target = input décalé de 1
    assert torch.equal(x[1:], y[:-1]), "Erreur: target n'est pas décalé correctement!"
    print("  ✅ Target = Input décalé de 1")
    
    # Décoder pour visualiser
    print(f"\n  Input décodé: '{tokenizer.decode(x.tolist())[:100]}...'")
    
    # Tester le DataLoader
    print("\n--- Test DataLoader ---")
    dataloader = create_dataloader(
        sample_text,
        tokenizer,
        seq_len=64,
        batch_size=4,
        shuffle=True
    )
    
    # Prendre un batch
    batch_x, batch_y = next(iter(dataloader))
    print(f"Batch input shape: {batch_x.shape}")   # (4, 64)
    print(f"Batch target shape: {batch_y.shape}")  # (4, 64)
    
    print("\n✅ Dataset et DataLoader fonctionnent!")