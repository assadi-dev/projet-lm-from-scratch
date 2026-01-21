---
name: llm-from-scratch
description: Guide complet pour développer un petit LLM (Large Language Model) de génération de texte from scratch. Utiliser ce skill pour (1) Construire l'architecture Transformer decoder-only, (2) Préparer et tokeniser des données textuelles, (3) Entraîner un modèle de langage, (4) Générer du texte avec différentes stratégies de sampling, (5) Comprendre chaque composant d'un LLM. Cible l'environnement Antigravity avec Gemini. Compatible Windows.
---

# LLM From Scratch - Guide de Développement

## Vue d'ensemble

Ce skill guide la création d'un petit LLM de génération de texte en PyTorch, optimisé pour Windows avec GPU NVIDIA.

## Prérequis Windows

### Installation de Python

1. Télécharger Python 3.10+ depuis [python.org](https://www.python.org/downloads/)
2. **Cocher "Add Python to PATH"** lors de l'installation

### Installation de CUDA (pour GPU NVIDIA)

1. Vérifier ta carte graphique : `nvidia-smi` dans CMD
2. Télécharger CUDA Toolkit depuis [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads)
3. Installer cuDNN correspondant

### Installation de PyTorch avec CUDA

```powershell
# Dans PowerShell ou CMD
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Vérifier l'installation :
```python
import torch
print(torch.cuda.is_available())  # Doit afficher True
print(torch.cuda.get_device_name(0))  # Nom de ta carte
```

## Structure du projet

```
llm-project\
├── config\
│   └── config.yaml          # Hyperparamètres
├── data\
│   ├── raw\                  # Données brutes
│   └── processed\            # Données tokenisées
├── src\
│   ├── __init__.py
│   ├── model\
│   │   ├── __init__.py
│   │   ├── transformer.py    # Architecture complète
│   │   ├── attention.py      # Multi-head attention
│   │   ├── layers.py         # Feed-forward, LayerNorm
│   │   └── embeddings.py     # Token + positional embeddings
│   ├── data\
│   │   ├── __init__.py
│   │   ├── tokenizer.py      # Tokenization BPE
│   │   └── dataset.py        # DataLoader PyTorch
│   ├── training\
│   │   ├── __init__.py
│   │   ├── trainer.py        # Boucle d'entraînement
│   │   └── scheduler.py      # Learning rate scheduling
│   └── inference\
│       ├── __init__.py
│       └── generate.py       # Génération de texte
├── scripts\
│   ├── prepare_data.py       # Préparation des données
│   ├── train.py              # Script d'entraînement
│   └── generate.py           # Script de génération
├── checkpoints\              # Modèles sauvegardés
├── logs\                     # Logs d'entraînement
└── requirements.txt
```

## Démarrage rapide (Windows)

### Option 1: PowerShell

```powershell
# 1. Créer le projet
.\scripts\setup_project.ps1 mon-llm

# 2. Aller dans le projet
cd mon-llm

# 3. Créer l'environnement virtuel
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Installer PyTorch avec CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option 2: CMD

```batch
REM 1. Créer le projet
scripts\setup_project.bat mon-llm

REM 2. Aller dans le projet
cd mon-llm

REM 3. Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate.bat

REM 4. Installer les dépendances
pip install -r requirements.txt
```

### Télécharger des données

```powershell
# TinyStories (recommandé pour commencer, ~500 Mo)
python scripts\download_data.py --dataset tinystories --output data\raw\

# Ou Wikipedia français
python scripts\download_data.py --dataset wikipedia --lang fr --output data\raw\ --max-samples 10000
```

### Entraîner le tokenizer

```powershell
python scripts\train_tokenizer.py --input data\raw\ --output tokenizer.json --vocab-size 32000
```

## Workflow de développement

### Étape 1: Setup environnement ✓

Voir section "Démarrage rapide" ci-dessus.

### Étape 2: Préparer les données

Voir `references/data-preparation.md` pour le guide complet de tokenization.

### Étape 3: Implémenter le modèle

Voir `references/architecture.md` pour les détails de chaque composant.

Ordre d'implémentation recommandé:
1. `embeddings.py` - Embeddings + encodage positionnel
2. `attention.py` - Multi-head self-attention
3. `layers.py` - Feed-forward + normalisation
4. `transformer.py` - Assembler le modèle complet

**Raccourci**: Copier le template prêt à l'emploi :
```powershell
copy assets\model_template.py src\model\transformer.py
copy assets\trainer_template.py src\training\trainer.py
```

### Étape 4: Entraînement

Voir `references/training.md` pour les stratégies d'optimisation.

### Étape 5: Génération

Voir `references/generation.md` pour les méthodes de sampling.

## Hyperparamètres recommandés (petit modèle)

```yaml
# Pour ~15M paramètres, entraînable sur GPU 8Go
model:
  vocab_size: 32000
  d_model: 256
  n_heads: 8
  n_layers: 6
  d_ff: 1024
  max_seq_len: 512
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 3e-4
  warmup_steps: 1000
  max_steps: 100000
  gradient_clip: 1.0
```

## Problèmes courants Windows

### "CUDA out of memory"

Réduire le batch_size dans config.yaml :
```yaml
training:
  batch_size: 16  # ou 8
  gradient_accumulation_steps: 8  # compenser
```

### "torch.cuda.is_available() returns False"

1. Vérifier que CUDA est installé : `nvcc --version`
2. Réinstaller PyTorch avec CUDA :
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Erreur d'encodage UTF-8

Ajouter en haut de tes scripts Python :
```python
# -*- coding: utf-8 -*-
```

Ou dans PowerShell :
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

## Scripts utilitaires

- `scripts\setup_project.ps1` / `.bat` - Initialise la structure
- `scripts\download_data.py` - Télécharge des datasets
- `scripts\train_tokenizer.py` - Entraîne un tokenizer BPE

## Ressources de référence

- `references/architecture.md` - Détails de l'architecture Transformer
- `references/data-preparation.md` - Guide de préparation des données
- `references/training.md` - Stratégies d'entraînement
- `references/generation.md` - Méthodes de génération de texte
- `references/troubleshooting.md` - Problèmes courants et solutions
