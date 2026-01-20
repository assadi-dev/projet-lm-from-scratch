# 🧠 LLM From Scratch : Projet de Modèle de Langage (Architecture Transformer)

Ce projet vise à implémenter entièrement un modèle de langage (LLM) de type **Decoder-only** (similaire à GPT) à partir de zéro, en utilisant **PyTorch**.

## 🚀 Vue d'ensemble

Le projet couvre toute la chaîne de production d'un LLM :
- **Architecture** : Implémentation complète des blocs Transformer (Attention, Feed-Forward, Normalisation).
- **Tokenisation** : Gestion des différents types de tokeniseurs.
- **Entraînement** : Scripts pour le pré-entraînement et le fine-tuning.
- **Inférence** : Génération de texte optimisée utilisant le masquage causal.

## 🏗️ Architecture Technique

Le modèle suit une architecture **Decoder-only Transformer** moderne, incluant :

- **Embeddings** : Support pour les embeddings de tokens et l'encodage positionnel (Sinusoïdal ou RoPE).
- **Multi-Head Self-Attention** : Implémentation avec masque causal pour empêcher le modèle de regarder les tokens futurs.
- **Feed-Forward Blocks** : Utilisation de GELU ou SwiGLU pour une meilleure convergence.
- **Pre-LayerNorm** : Normalisation avant chaque sous-bloc (recommandé pour la stabilité de l'entraînement).
- **Inference** : Mécanismes de génération auto-régressive.

## 📁 Structure du Projet

```text
.
├── checkpoints/       # Sauvegarde des poids du modèle
├── config/            # Fichiers de configuration YAML
├── data/              # Dossiers pour les données brutes et traitées
├── logs/              # Logs de Tensorboard et W&B
├── notebooks/         # Expérimentations interactives
├── scripts/           # Utilitaires de traitement de données
├── src/               # Code source principal
│   ├── model/         # Définition de l'architecture Transformer
│   ├── data/          # Chargeurs de données (DataLoaders)
│   ├── training/      # Boucles d'entraînement
│   └── inference/     # Scripts de génération de texte
├── README.md          # Documentation principale
├── architecture.md    # Guide technique détaillé
├── requirements.txt   # Dépendances Python
└── setup_project.bat  # Script d'initialisation
```

## 🛠️ Installation et Configuration

### Pré-requis
- Python 3.8+
- PyTorch 2.0+

### Installation (Windows)

1. **Initialiser la structure** (si ce n'est pas déjà fait) :
   ```bash
   setup_project.bat
   ```

2. **Créer l'environnement virtuel** :
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Utilisation

1. **Préparation des données** : Placez vos fichiers `.txt` dans `data/raw/`.
2. **Entraînement** : Utilisez les scripts dans `src/training/`.
3. **Génération** : Testez le modèle avec les scripts dans `src/inference/`.
4. **Chat Interactif** : Démarrez une session de chat avec le modèle :
   ```bash
   python src/chat.py
   ```

## 📚 Ressources
- `architecture.md` : Guide détaillé sur les mathématiques et l'implémentation des composants.
- `model_template.py` : Modèle de base pour l'implémentation.

---
*Projet développé dans le cadre d'un apprentissage approfondi des Transformers.*
