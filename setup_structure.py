import os
from pathlib import Path

# Definição da estrutura de pastas
project_structure = [
    "config",
    "data/raw",
    "data/processed",
    "data/external",
    "docs",
    "models",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/visualization",
    "tests",
    ".github/workflows",  # Para CI/CD futuro
]

# Arquivos que devem ser criados
files_to_create = [
    "params.yaml",         # Para hiperparâmetros
    ".gitignore",          # Caso não exista
    "src/__init__.py",     # Torna src um pacote Python
    "src/data/__init__.py",
    "src/features/__init__.py",
    "src/models/__init__.py",
    "src/visualization/__init__.py",
]

def create_structure():
    root_path = Path.cwd()
    print(f"🔨 Criando estrutura em: {root_path}")

    # 1. Criar pastas
    for folder in project_structure:
        path = root_path / folder
        path.mkdir(parents=True, exist_ok=True)
        # Cria um .gitkeep para o Git rastrear pastas vazias
        (path / ".gitkeep").touch()
        print(f"✅ Pasta criada: {folder}")

    # 2. Criar arquivos iniciais
    for file in files_to_create:
        path = root_path / file
        if not path.exists():
            path.touch()
            print(f"📄 Arquivo criado: {file}")
        else:
            print(f"⏩ Arquivo já existe: {file}")

if __name__ == "__main__":
    create_structure()