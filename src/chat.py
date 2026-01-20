"""
chat.py - Interface terminal pour discuter avec ton LLM

Usage:
    python chat.py                     # Charge le modèle français par défaut
    python chat.py --model english     # Charge le modèle anglais
    python chat.py --model chemin/vers/model.pt  # Charge un modèle spécifique
"""

import torch
import os
import sys
import argparse
from pathlib import Path

# Ajouter les chemins
src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_path)
sys.path.append(os.path.join(src_path, "model"))
sys.path.append(os.path.join(src_path, "data"))
sys.path.append(os.path.join(src_path, "inference"))

from gpt import GPT
from tokenizer import Tokenizer
from generate import TextGenerator


class TerminalChat:
    """
    Interface de chat dans le terminal.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = Tokenizer()
        
        # Charger le modèle
        print(f"🔄 Chargement du modèle: {model_path}")
        
        self.model = GPT(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            max_seq_len=256,
            dropout=0.0  # Pas de dropout en inférence
        )
        
        # Charger les poids
        state_dict = torch.load(model_path, map_location=device)
        
        # Gérer les deux formats de sauvegarde
        if 'model_state_dict' in state_dict:
            self.model.load_state_dict(state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(state_dict)
        
        self.model.to(device)
        self.model.eval()
        
        # Générateur
        self.generator = TextGenerator(self.model, self.tokenizer, device)
        
        # Paramètres par défaut
        self.temperature = 0.8
        self.top_k = 50
        self.max_tokens = 100
        
        print("✅ Modèle chargé!")
    
    def generate(self, prompt: str) -> str:
        """Génère du texte à partir d'un prompt."""
        return self.generator.generate(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k
        )
    
    def generate_stream(self, prompt: str):
        """Génère du texte en streaming (token par token)."""
        return self.generator.generate_stream(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k
        )
    
    def print_help(self):
        """Affiche l'aide."""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║                    🤖 COMMANDES DISPONIBLES                  ║
╠══════════════════════════════════════════════════════════════╣
║  /help           - Afficher cette aide                       ║
║  /temp <valeur>  - Changer la température (ex: /temp 0.9)    ║
║  /topk <valeur>  - Changer le top_k (ex: /topk 40)           ║
║  /tokens <n>     - Changer le nombre de tokens (ex: /tokens 150) ║
║  /settings       - Afficher les paramètres actuels           ║
║  /stream         - Activer/désactiver le mode streaming      ║
║  /clear          - Effacer l'écran                           ║
║  /quit           - Quitter                                   ║
╠══════════════════════════════════════════════════════════════╣
║  💡 Sinon, tape simplement ton texte et appuie sur Entrée    ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(help_text)
    
    def print_settings(self):
        """Affiche les paramètres actuels."""
        print(f"""
┌─────────────────────────────────┐
│     ⚙️  PARAMÈTRES ACTUELS      │
├─────────────────────────────────┤
│  Température : {self.temperature:<16} │
│  Top-K       : {self.top_k:<16} │
│  Max tokens  : {self.max_tokens:<16} │
│  Device      : {self.device:<16} │
└─────────────────────────────────┘
""")
    
    def run(self):
        """Lance l'interface de chat."""
        
        # En-tête
        print("\n" + "=" * 60)
        print("🤖 LLM FROM SCRATCH - MODE TERMINAL")
        print("=" * 60)
        print("Tape /help pour voir les commandes disponibles.")
        print("Tape /quit pour quitter.")
        print("=" * 60 + "\n")
        
        streaming = False
        
        while True:
            try:
                # Prompt utilisateur
                user_input = input("📝 Toi: ").strip()
                
                # Ignorer les entrées vides
                if not user_input:
                    continue
                
                # Commandes spéciales
                if user_input.startswith("/"):
                    parts = user_input.split()
                    cmd = parts[0].lower()
                    
                    if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                        print("\n👋 À bientôt!")
                        break
                    
                    elif cmd == "/help":
                        self.print_help()
                    
                    elif cmd == "/settings":
                        self.print_settings()
                    
                    elif cmd == "/clear":
                        os.system('cls' if os.name == 'nt' else 'clear')
                    
                    elif cmd == "/stream":
                        streaming = not streaming
                        status = "activé" if streaming else "désactivé"
                        print(f"  Mode streaming {status}")
                    
                    elif cmd == "/temp" and len(parts) > 1:
                        try:
                            self.temperature = float(parts[1])
                            print(f"  Température → {self.temperature}")
                        except:
                            print("  ❌ Valeur invalide (ex: /temp 0.8)")
                    
                    elif cmd == "/topk" and len(parts) > 1:
                        try:
                            self.top_k = int(parts[1])
                            print(f"  Top-K → {self.top_k}")
                        except:
                            print("  ❌ Valeur invalide (ex: /topk 50)")
                    
                    elif cmd == "/tokens" and len(parts) > 1:
                        try:
                            self.max_tokens = int(parts[1])
                            print(f"  Max tokens → {self.max_tokens}")
                        except:
                            print("  ❌ Valeur invalide (ex: /tokens 100)")
                    
                    else:
                        print("  ❌ Commande inconnue. Tape /help pour l'aide.")
                    
                    continue
                
                # Génération
                print("\n🤖 LLM: ", end="", flush=True)
                
                if streaming:
                    # Mode streaming (token par token)
                    first = True
                    for token in self.generate_stream(user_input):
                        if first:
                            first = False
                            continue  # Skip le prompt
                        print(token, end="", flush=True)
                    print("\n")
                else:
                    # Mode normal
                    output = self.generate(user_input)
                    print(output + "\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 À bientôt!")
                break
            
            except Exception as e:
                print(f"\n❌ Erreur: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Chat avec ton LLM")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="french",
        help="Modèle à charger: 'french', 'english', ou chemin vers un fichier .pt"
    )
    args = parser.parse_args()
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Trouver le modèle
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    
    if args.model == "french":
        model_path = checkpoint_dir / "best_model_french.pt"
    elif args.model == "english":
        model_path = checkpoint_dir / "best_model.pt"
    else:
        model_path = Path(args.model)
    
    # Vérifier que le modèle existe
    if not model_path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        print("\nModèles disponibles:")
        for f in checkpoint_dir.glob("*.pt"):
            print(f"  - {f.name}")
        print("\nUsage:")
        print("  python chat.py --model french")
        print("  python chat.py --model english")
        print("  python chat.py --model chemin/vers/model.pt")
        return
    
    # Lancer le chat
    chat = TerminalChat(str(model_path), device)
    chat.run()


if __name__ == "__main__":
    main()