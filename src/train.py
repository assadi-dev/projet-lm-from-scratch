"""
train.py - Script principal d'entraînement

Entraîne un modèle GPT sur le dataset TinyStories.
"""

import torch
import os
import sys
import time
from pathlib import Path

# Ajouter les chemins
src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "model"))
sys.path.append(os.path.join(src_path, "data"))
sys.path.append(os.path.join(src_path, "training"))
sys.path.append(os.path.join(src_path, "inference"))

from gpt import GPT
from tokenizer import Tokenizer
from dataset import create_dataloader
from trainer import Trainer
from generate import TextGenerator


def download_tinystories(max_samples: int = 10000) -> str:
    """
    Télécharge le dataset TinyStories.
    
    Args:
        max_samples: Nombre maximum d'histoires à télécharger
    
    Returns:
        Le texte complet
    """
    print("=" * 60)
    print("TÉLÉCHARGEMENT DU DATASET")
    print("=" * 60)
    
    from datasets import load_dataset
    
    print("Chargement de TinyStories depuis HuggingFace...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # Limiter le nombre d'échantillons
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    print(f"Nombre d'histoires: {len(dataset):,}")
    
    # Concaténer toutes les histoires
    texts = []
    for example in dataset:
        texts.append(example['text'])
    
    full_text = "\n\n".join(texts)
    print(f"Taille totale: {len(full_text):,} caractères")
    
    return full_text


def main():
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Dataset
    MAX_SAMPLES = 5000          # Nombre d'histoires (augmenter pour mieux apprendre)
    
    # Modèle
    D_MODEL = 256               # Dimension du modèle
    N_HEADS = 8                 # Têtes d'attention
    N_LAYERS = 6                # Blocs Transformer
    D_FF = 1024                 # Dimension feed-forward
    MAX_SEQ_LEN = 256           # Longueur max des séquences
    DROPOUT = 0.1
    
    # Entraînement
    BATCH_SIZE = 8              # Réduire si out of memory
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 3              # Nombre d'époques
    
    # Chemins
    CHECKPOINT_DIR = Path("../checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Mémoire: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # =========================================================================
    # PRÉPARATION DES DONNÉES
    # =========================================================================
    
    # Télécharger le dataset
    text = download_tinystories(max_samples=MAX_SAMPLES)
    
    # Créer le tokenizer
    print("\n" + "=" * 60)
    print("PRÉPARATION DES DONNÉES")
    print("=" * 60)
    
    tokenizer = Tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    
    # Créer les DataLoaders
    print("\nCréation du DataLoader...")
    
    # Split train/val (90/10)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    train_loader = create_dataloader(
        train_text,
        tokenizer,
        seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        val_text,
        tokenizer,
        seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    # =========================================================================
    # CRÉATION DU MODÈLE
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("CRÉATION DU MODÈLE")
    print("=" * 60)
    
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    )
    
    # =========================================================================
    # ENTRAÎNEMENT
    # =========================================================================
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    # Entraîner
    save_path = CHECKPOINT_DIR / "best_model.pt"
    trainer.train(num_epochs=NUM_EPOCHS, save_path=str(save_path))
    
    # Sauvegarder le checkpoint final
    final_path = CHECKPOINT_DIR / "final_model.pt"
    trainer.save_checkpoint(str(final_path))
    
    # =========================================================================
    # TEST DE GÉNÉRATION
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("TEST DE GÉNÉRATION")
    print("=" * 60)
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    generator = TextGenerator(model, tokenizer, device)
    
    # Tester plusieurs prompts
    prompts = [
        "Once upon a time",
        "The little girl",
        "One day, a boy",
        "There was a cat",
    ]
    
    for prompt in prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        print("-" * 40)
        
        output = generator.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50
        )
        print(output)
    
    # =========================================================================
    # RÉSUMÉ
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("=" * 60)
    print(f"Modèle sauvegardé: {save_path}")
    print(f"Checkpoint final: {final_path}")
    print(f"\nPour générer du texte plus tard:")
    print(f"  python inference/generate.py --model {save_path}")


if __name__ == "__main__":
    main()