"""
trainer.py - Boucle d'entraînement pour le LLM

Gère l'entraînement complet : forward, backward, optimizer, logging.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import time
import math


class Trainer:
    """
    Classe pour entraîner le modèle GPT.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        learning_rate: float = 3e-4,
        device: str = "cuda"
    ):
        """
        Args:
            model: Le modèle GPT à entraîner
            train_loader: DataLoader pour l'entraînement
            val_loader: DataLoader pour la validation (optionnel)
            learning_rate: Taux d'apprentissage
            device: "cuda" ou "cpu"
        """
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer (AdamW est standard pour les LLMs)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # Pour le logging
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """
        Entraîne le modèle pour une époque complète.
        
        Returns:
            Loss moyenne de l'époque
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            # Déplacer vers GPU
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            logits, loss = self.model(x, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (évite les explosions de gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update des poids
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            
            # Afficher la progression tous les 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Time: {elapsed:.1f}s")
        
        epoch_loss = total_loss / num_batches
        self.train_losses.append(epoch_loss)
        
        return epoch_loss
    
    @torch.no_grad()
    def evaluate(self):
        """
        Évalue le modèle sur le validation set.
        
        Returns:
            Loss moyenne de validation
        """
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            logits, loss = self.model(x, y)
            
            total_loss += loss.item()
            num_batches += 1
        
        val_loss = total_loss / num_batches
        self.val_losses.append(val_loss)
        
        return val_loss
    
    def train(self, num_epochs: int, save_path: str = None):
        """
        Entraîne le modèle pour plusieurs époques.
        
        Args:
            num_epochs: Nombre d'époques
            save_path: Chemin pour sauvegarder le meilleur modèle
        """
        print("=" * 60)
        print("DÉBUT DE L'ENTRAÎNEMENT")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Époques: {num_epochs}")
        print(f"Batches par époque: {len(self.train_loader)}")
        print("=" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n>>> Époque {epoch + 1}/{num_epochs}")
            
            # Entraînement
            train_loss = self.train_epoch()
            print(f"  Train Loss: {train_loss:.4f}")
            
            # Validation
            val_loss = self.evaluate()
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
                
                # Sauvegarder le meilleur modèle
                if save_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  ✅ Modèle sauvegardé (val_loss: {val_loss:.4f})")
            
            # Perplexité (métrique standard pour les LLMs)
            perplexity = math.exp(train_loss)
            print(f"  Perplexité: {perplexity:.2f}")
        
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT TERMINÉ")
        print("=" * 60)
    
    def save_checkpoint(self, path: str):
        """Sauvegarde complète (modèle + optimizer)"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint sauvegardé: {path}")
    
    def load_checkpoint(self, path: str):
        """Charge un checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Checkpoint chargé: {path}")


# =============================================================================
# TEST : Entraînement rapide sur données synthétiques
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    
    # Ajouter le dossier src au path
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(src_path)
    sys.path.append(os.path.join(src_path, "model"))
    sys.path.append(os.path.join(src_path, "data"))
    
    from model.gpt import GPT
    from data.tokenizer import Tokenizer
    from data.dataset import create_dataloader
    
    # Vérifier si CUDA est disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Créer le tokenizer
    tokenizer = Tokenizer()
    
    # Texte d'exemple (petit pour le test)
    sample_text = """
    Le chat dort sur le canapé. Il rêve de souris et de poissons.
    La nuit, le chat chasse dans le jardin. Il est très agile.
    Le matin, le chat mange ses croquettes. Il boit du lait frais.
    L'après-midi, le chat fait sa toilette. Il se lèche les pattes.
    Le soir, le chat joue avec une balle. Il saute très haut.
    """ * 50  # Répéter pour avoir assez de données
    
    # Paramètres réduits pour le test
    seq_len = 64
    batch_size = 4
    vocab_size = tokenizer.vocab_size  # 50257 pour GPT-2
    
    # Créer le DataLoader
    train_loader = create_dataloader(
        sample_text,
        tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Créer un petit modèle pour le test
    model = GPT(
        vocab_size=vocab_size,
        d_model=128,      # Petit pour le test
        n_heads=4,
        n_layers=2,       # Peu de couches
        d_ff=256,
        max_seq_len=seq_len
    )
    
    # Créer le trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        learning_rate=1e-3,
        device=device
    )
    
    # Entraîner pour 2 époques (test rapide)
    print("\n--- Test d'entraînement (2 époques) ---")
    trainer.train(num_epochs=2)
    
    print("\n✅ Trainer fonctionne!")