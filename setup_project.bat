@echo off
REM setup_project.bat - Initialise la structure du projet LLM (Windows CMD)

set PROJECT_NAME=%1
if "%PROJECT_NAME%"=="" set PROJECT_NAME=.

echo 🚀 Creation du projet: %PROJECT_NAME%

REM Creer la structure des dossiers
mkdir "%PROJECT_NAME%\config" 2>nul
mkdir "%PROJECT_NAME%\data\raw" 2>nul
mkdir "%PROJECT_NAME%\data\processed" 2>nul
mkdir "%PROJECT_NAME%\src\model" 2>nul
mkdir "%PROJECT_NAME%\src\data" 2>nul
mkdir "%PROJECT_NAME%\src\training" 2>nul
mkdir "%PROJECT_NAME%\src\inference" 2>nul
mkdir "%PROJECT_NAME%\scripts" 2>nul
mkdir "%PROJECT_NAME%\checkpoints" 2>nul
mkdir "%PROJECT_NAME%\logs" 2>nul
mkdir "%PROJECT_NAME%\notebooks" 2>nul

REM Creer les fichiers __init__.py
type nul > "%PROJECT_NAME%\src\__init__.py"
type nul > "%PROJECT_NAME%\src\model\__init__.py"
type nul > "%PROJECT_NAME%\src\data\__init__.py"
type nul > "%PROJECT_NAME%\src\training\__init__.py"
type nul > "%PROJECT_NAME%\src\inference\__init__.py"

REM Creer requirements.txt
(
echo torch^>=2.0.0
echo numpy^>=1.24.0
echo tokenizers^>=0.15.0
echo tiktoken^>=0.5.0
echo sentencepiece^>=0.1.99
echo datasets^>=2.14.0
echo transformers^>=4.35.0
echo wandb^>=0.15.0
echo tensorboard^>=2.14.0
echo tqdm^>=4.66.0
echo pyyaml^>=6.0
echo einops^>=0.7.0
echo matplotlib^>=3.7.0
) > "%PROJECT_NAME%\requirements.txt"

echo ✅ Structure creee!
echo.
echo Prochaines etapes:
echo 1. cd %PROJECT_NAME%
echo 2. python -m venv venv
echo 3. venv\Scripts\activate.bat
echo 4. pip install -r requirements.txt
echo 5. Placer vos donnees dans data\raw\
echo.

dir /s /b /ad "%PROJECT_NAME%"
