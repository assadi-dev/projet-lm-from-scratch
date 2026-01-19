# setup_project.ps1 - Initialise la structure du projet LLM (Windows PowerShell)

param(
    [string]$ProjectName = "llm-project"
)

Write-Host "🚀 Création du projet: $ProjectName" -ForegroundColor Cyan

# Créer la structure des dossiers
$folders = @(
    "$ProjectName/config",
    "$ProjectName/data/raw",
    "$ProjectName/data/processed",
    "$ProjectName/src/model",
    "$ProjectName/src/data",
    "$ProjectName/src/training",
    "$ProjectName/src/inference",
    "$ProjectName/scripts",
    "$ProjectName/checkpoints",
    "$ProjectName/logs",
    "$ProjectName/notebooks"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}

# Créer les fichiers __init__.py
$initFiles = @(
    "$ProjectName/src/__init__.py",
    "$ProjectName/src/model/__init__.py",
    "$ProjectName/src/data/__init__.py",
    "$ProjectName/src/training/__init__.py",
    "$ProjectName/src/inference/__init__.py"
)

foreach ($file in $initFiles) {
    New-Item -ItemType File -Force -Path $file | Out-Null
}

# Créer le fichier de configuration
$configContent = @"
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
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4
  min_lr: 3.0e-5
  warmup_steps: 1000
  max_steps: 100000
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  gradient_clip: 1.0
  eval_interval: 500
  eval_steps: 100
  save_interval: 1000
  checkpoint_dir: "checkpoints"

data:
  train_path: "data/processed/train_tokens.pt"
  val_path: "data/processed/val_tokens.pt"
  seq_length: 512

system:
  device: "cuda"
  dtype: "bfloat16"
  compile: true
  seed: 42
"@

$configContent | Out-File -FilePath "$ProjectName/config/config.yaml" -Encoding UTF8

# Créer requirements.txt
$requirementsContent = @"
torch>=2.0.0
numpy>=1.24.0
tokenizers>=0.15.0
tiktoken>=0.5.0
sentencepiece>=0.1.99
datasets>=2.14.0
transformers>=4.35.0
wandb>=0.15.0
tensorboard>=2.14.0
tqdm>=4.66.0
pyyaml>=6.0
einops>=0.7.0
matplotlib>=3.7.0
"@

$requirementsContent | Out-File -FilePath "$ProjectName/requirements.txt" -Encoding UTF8

# Créer .gitignore
$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*`$py.class
*.so
.Python
venv/
env/
.env

# Data
data/raw/
data/processed/
*.pt
*.bin

# Checkpoints
checkpoints/
*.ckpt

# Logs
logs/
wandb/
runs/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS
.DS_Store
Thumbs.db
"@

$gitignoreContent | Out-File -FilePath "$ProjectName/.gitignore" -Encoding UTF8

Write-Host "✅ Structure créée!" -ForegroundColor Green
Write-Host ""
Write-Host "Prochaines étapes:" -ForegroundColor Yellow
Write-Host "1. cd $ProjectName"
Write-Host "2. python -m venv venv"
Write-Host "3. .\venv\Scripts\Activate.ps1"
Write-Host "4. pip install -r requirements.txt"
Write-Host "5. Placer vos données dans data\raw\"
Write-Host ""

# Afficher la structure
Write-Host "Structure créée:" -ForegroundColor Cyan
Get-ChildItem -Path $ProjectName -Recurse -Directory | ForEach-Object {
    $indent = "  " * ($_.FullName.Split("\").Count - $ProjectName.Split("\").Count)
    Write-Host "$indent$($_.Name)/"
}
