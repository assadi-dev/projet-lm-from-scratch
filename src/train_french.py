"""
train_french.py - Entraîner le LLM sur un dataset français

Télécharge automatiquement un dataset français depuis HuggingFace.
"""

import torch
import os
import sys
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


def download_french_wikipedia(max_samples: int = 5000) -> str:
    """
    Télécharge des articles Wikipedia en français.
    
    Args:
        max_samples: Nombre d'articles à télécharger
    
    Returns:
        Texte concaténé
    """
    print("=" * 60)
    print("TÉLÉCHARGEMENT WIKIPEDIA FRANÇAIS")
    print("=" * 60)
    
    from datasets import load_dataset
    
    print("Chargement du dataset (peut prendre quelques minutes)...")
    
    # Charger Wikipedia français en streaming (évite de tout télécharger)
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.fr",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    texts = []
    count = 0
    
    print(f"Extraction de {max_samples} articles...")
    
    for article in dataset:
        if count >= max_samples:
            break
        
        text = article['text']
        
        # Garder seulement les articles assez longs
        if len(text) > 1000:
            # Prendre les premiers paragraphes (éviter les articles trop longs)
            text = text[:3000]
            texts.append(text)
            count += 1
            
            if count % 500 == 0:
                print(f"  {count}/{max_samples} articles chargés...")
    
    full_text = "\n\n".join(texts)
    print(f"\n✅ {count} articles chargés!")
    print(f"   Taille totale: {len(full_text):,} caractères")
    
    return full_text


def download_french_books(max_samples: int = 5000) -> str:
    """
    Télécharge des textes littéraires français (Gutenberg).
    """
    print("=" * 60)
    print("TÉLÉCHARGEMENT TEXTES FRANÇAIS")
    print("=" * 60)
    
    from datasets import load_dataset
    
    print("Chargement du dataset...")
    
    # Dataset de textes français variés
    dataset = load_dataset(
        "uonlp/CulturaX",
        "fr",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    texts = []
    count = 0
    
    print(f"Extraction de {max_samples} textes...")
    
    for item in dataset:
        if count >= max_samples:
            break
        
        text = item['text']
        
        # Filtrer les textes de bonne qualité
        if len(text) > 500 and len(text) < 5000:
            texts.append(text)
            count += 1
            
            if count % 500 == 0:
                print(f"  {count}/{max_samples} textes chargés...")
    
    full_text = "\n\n".join(texts)
    print(f"\n✅ {count} textes chargés!")
    print(f"   Taille totale: {len(full_text):,} caractères")
    
    return full_text


def download_french_stories(max_samples: int = 3000) -> str:
    """
    Télécharge des histoires/contes en français.
    Alternative plus légère.
    """
    print("=" * 60)
    print("TÉLÉCHARGEMENT HISTOIRES FRANÇAISES")
    print("=" * 60)
    
    from datasets import load_dataset
    
    print("Chargement du dataset...")
    
    # Utiliser un dataset de textes français plus léger
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        "fr",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    texts = []
    count = 0
    
    print(f"Extraction de {max_samples} textes...")
    
    for item in dataset:
        if count >= max_samples:
            break
        
        text = item['text']
        
        # Filtrer : garder les textes de taille moyenne
        if 200 < len(text) < 2000:
            texts.append(text)
            count += 1
            
            if count % 500 == 0:
                print(f"  {count}/{max_samples} textes chargés...")
    
    full_text = "\n\n".join(texts)
    print(f"\n✅ {count} textes chargés!")
    print(f"   Taille totale: {len(full_text):,} caractères")
    
    return full_text


def main():
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    print("=" * 60)
    print("🇫🇷 ENTRAÎNEMENT LLM FRANÇAIS")
    print("=" * 60)
    
    print("\nChoisir le dataset:")
    print("1. Wikipedia français (recommandé, texte de qualité)")
    print("2. OSCAR français (texte web varié)")
    print("3. CulturaX français (textes littéraires)")
    
    choice = input("\nTon choix (1, 2 ou 3): ").strip()
    
    # Nombre d'échantillons
    print("\nCombien de textes télécharger?")
    print("  - 1000  : Test rapide (~5 min d'entraînement)")
    print("  - 5000  : Bon résultat (~15 min)")
    print("  - 10000 : Meilleur résultat (~30 min)")
    
    try:
        max_samples = int(input("\nNombre de textes (défaut 5000): ").strip() or "5000")
    except:
        max_samples = 5000
    
    # =========================================================================
    # PARAMÈTRES DU MODÈLE
    # =========================================================================
    
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 6
    D_FF = 1024
    MAX_SEQ_LEN = 256
    DROPOUT = 0.1
    
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 5
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # =========================================================================
    # TÉLÉCHARGEMENT DES DONNÉES
    # =========================================================================
    
    if choice == "2":
        text = download_french_stories(max_samples)
    elif choice == "3":
        text = download_french_books(max_samples)
    else:
        text = download_french_wikipedia(max_samples)
    
    # Sauvegarder le texte pour réutilisation
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    
    data_file = data_dir / "french_text.txt"
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n💾 Données sauvegardées: {data_file}")
    
    # =========================================================================
    # PRÉPARATION
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PRÉPARATION DES DONNÉES")
    print("=" * 60)
    
    tokenizer = Tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    
    # Split train/val (90/10)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    train_loader = create_dataloader(
        train_text, tokenizer, seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = create_dataloader(
        val_text, tokenizer, seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE, shuffle=False
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
    
    CHECKPOINT_DIR = Path("../checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    save_path = CHECKPOINT_DIR / "best_model_french.pt"
    trainer.train(num_epochs=NUM_EPOCHS, save_path=str(save_path))
    
    # Sauvegarder le checkpoint final
    final_path = CHECKPOINT_DIR / "final_model_french.pt"
    trainer.save_checkpoint(str(final_path))
    
    # =========================================================================
    # TEST DE GÉNÉRATION
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("🇫🇷 TEST DE GÉNÉRATION EN FRANÇAIS")
    print("=" * 60)
    
    model.load_state_dict(torch.load(save_path, map_location=device))
    generator = TextGenerator(model, tokenizer, device)
    
    prompts_fr = [
        "La France est",
        "Il était une fois",
        "Le président",
        "Dans la ville de Paris",
        "L'histoire de",
        "Les enfants",
    ]
    
    for prompt in prompts_fr:
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
    # MODE INTERACTIF
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("💬 MODE INTERACTIF")
    print("=" * 60)
    print("Tape un début de phrase et le modèle va continuer.")
    print("Tape 'quit' pour quitter.\n")
    
    while True:
        prompt = input("🇫🇷 Ton prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
        
        output = generator.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50
        )
        print(f"\n📖 {output}\n")
    
    print("\n✅ Terminé! Modèle sauvegardé:", save_path)


if __name__ == "__main__":
    main()