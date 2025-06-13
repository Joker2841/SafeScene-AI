#!/usr/bin/env python3
"""
SafeScene AI - Environment Setup Script
This script sets up the complete development environment for the project.
File: scripts/setup_environment.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class EnvironmentSetup:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.directories = [
            # Data directories
            "data/raw/cityscapes",
            "data/raw/kitti", 
            "data/raw/custom_edge_cases",
            "data/processed/train",
            "data/processed/val",
            "data/processed/test",
            "data/synthetic/generated_scenes",
            "data/synthetic/metadata",
            
            # Source directories
            "src/core/gan",
            "src/core/rl",
            "src/core/evaluation",
            "src/data_processing",
            "src/training",
            "src/utils",
            
            # Web interface directories
            "web_interface/backend/api",
            "web_interface/backend/services",
            "web_interface/frontend/css",
            "web_interface/frontend/js",
            "web_interface/frontend/assets/images",
            "web_interface/frontend/assets/demos",
            "web_interface/static/generated",
            
            # Experiment directories
            "experiments/notebooks",
            "experiments/configs",
            "experiments/results/checkpoints",
            "experiments/results/logs",
            "experiments/results/metrics",
            "experiments/results/generated_samples",
            
            # Other directories
            "scripts",
            "tests",
            "docs/images",
            "deployment/cloud",
            "presentation/demo_videos",
            "presentation/slides",
            "presentation/screenshots"
        ]
    
    def create_directory_structure(self):
        """Create all project directories"""
        print("📁 Creating directory structure...")
        for directory in self.directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py for Python packages
            if "src" in str(directory) and not any(x in str(directory) for x in ["assets", "static", "css", "js"]):
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("# SafeScene AI\n")
        
        print("✅ Directory structure created successfully!")
    
    def create_requirements_file(self):
        """Create requirements.txt file"""
        print("📝 Creating requirements.txt...")
        requirements_content = """# SafeScene AI - Python Requirements
# Core ML/AI Libraries
torch>=2.0.0
torchvision>=0.15.0
stable-baselines3>=2.0.0
transformers>=4.30.0

# GAN Implementation
lpips
pytorch-fid
kornia>=0.7.0

# Computer Vision
opencv-python>=4.8.0
Pillow>=9.5.0
scikit-image>=0.21.0
albumentations>=1.3.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
h5py>=3.9.0

# Web Interface
fastapi>=0.100.0
uvicorn>=0.22.0
streamlit>=1.25.0
plotly>=5.15.0
pydantic>=2.0.0

# Utilities
tqdm>=4.65.0
wandb>=0.15.0
tensorboard>=2.13.0
pyyaml>=6.0
python-dotenv>=1.0.0
hydra-core>=1.3.0

# Development
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
pre-commit>=3.3.0
"""
        
        req_file = self.project_root / "requirements.txt"
        req_file.write_text(requirements_content)
        print("✅ requirements.txt created!")
    
    def create_conda_environment(self):
        """Create conda environment.yml file"""
        print("📝 Creating environment.yml...")
        conda_content = """name: safescene-ai
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch>=2.0.0
  - torchvision>=0.15.0
  - pytorch-cuda=11.8
  - cudatoolkit=11.8
  - pip
  - jupyter
  - ipykernel
  - pip:
    - stable-baselines3>=2.0.0
    - fastapi>=0.100.0
    - streamlit>=1.25.0
    - wandb>=0.15.0
    - opencv-python>=4.8.0
    - pytorch-fid
    - lpips
    - kornia>=0.7.0
    - albumentations>=1.3.0
    - hydra-core>=1.3.0
"""
        
        env_file = self.project_root / "environment.yml"
        env_file.write_text(conda_content)
        print("✅ environment.yml created!")
    
    def create_gitignore(self):
        """Create .gitignore file"""
        print("📝 Creating .gitignore...")
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Data
data/raw/
data/processed/
*.h5
*.hdf5
*.npy
*.npz

# Model checkpoints
*.pth
*.pt
*.ckpt
checkpoints/
models/

# Logs
logs/
runs/
wandb/
*.log

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local

# Generated files
generated/
outputs/
results/
*.png
*.jpg
*.jpeg
*.gif
*.mp4

# Documentation build
docs/_build/
site/

# Testing
.pytest_cache/
.coverage
htmlcov/
"""
        
        gitignore_file = self.project_root / ".gitignore"
        gitignore_file.write_text(gitignore_content)
        print("✅ .gitignore created!")
    
    def create_env_example(self):
        """Create .env.example file"""
        print("📝 Creating .env.example...")
        env_content = """# SafeScene AI Environment Variables

# Paths
DATA_ROOT=./data
CHECKPOINT_DIR=./experiments/results/checkpoints
LOG_DIR=./experiments/results/logs

# Training
BATCH_SIZE=32
LEARNING_RATE=0.0002
NUM_EPOCHS=100
DEVICE=cuda

# Weights & Biases (optional)
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=safescene-ai
WANDB_ENTITY=your_entity_here

# API
API_HOST=0.0.0.0
API_PORT=8000

# GPU
CUDA_VISIBLE_DEVICES=0
"""
        
        env_file = self.project_root / ".env.example"
        env_file.write_text(env_content)
        print("✅ .env.example created!")
    
    def setup_git_repository(self):
        """Initialize git repository"""
        print("🔧 Setting up Git repository...")
        try:
            # Check if git is already initialized
            if not (self.project_root / ".git").exists():
                subprocess.run(["git", "init"], cwd=self.project_root, check=True)
                subprocess.run(["git", "add", "."], cwd=self.project_root, check=True)
                subprocess.run(["git", "commit", "-m", "Initial commit: SafeScene AI project structure"], 
                             cwd=self.project_root, check=True)
                print("✅ Git repository initialized!")
            else:
                print("ℹ️  Git repository already exists")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git setup failed: {e}")
    
    def install_dependencies(self, use_conda=True):
        """Install project dependencies"""
        print("📦 Installing dependencies...")
        
        if use_conda:
            print("Using Conda for installation...")
            try:
                # Create conda environment
                subprocess.run(["conda", "env", "create", "-f", "environment.yml"], 
                             cwd=self.project_root, check=True)
                print("✅ Conda environment created!")
                print("\n🎯 To activate the environment, run:")
                print("   conda activate safescene-ai")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Conda installation failed: {e}")
                print("Falling back to pip...")
                self.install_with_pip()
        else:
            self.install_with_pip()
    
    def install_with_pip(self):
        """Install dependencies using pip"""
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         cwd=self.project_root, check=True)
            print("✅ Dependencies installed with pip!")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Pip installation failed: {e}")
    
    def create_readme(self):
        """Create initial README.md"""
        print("📝 Creating README.md...")
        readme_content = """# 🚀 SafeScene AI

**Intelligent Synthetic Data Generation for Autonomous Vehicle Safety**

## 🎯 Overview
SafeScene AI is a revolutionary AI system that uses reinforcement learning to intelligently guide GANs to generate challenging driving scenarios for autonomous vehicle training.

## 🏆 Key Features
- **RL-Guided Generation**: Smart scenario creation based on training needs
- **Multi-Modal Synthesis**: Weather, lighting, and object variations
- **Real-Time Interface**: Professional web application for generation control
- **Measurable Impact**: Proven improvement in AV safety metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 50GB+ free disk space

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/safescene-ai.git
cd safescene-ai

# Run setup script
python scripts/setup_environment.py --install

# Activate environment
conda activate safescene-ai
```

### Download Datasets
```bash
python scripts/download_datasets.py --dataset cityscapes --dataset kitti
```

### Train Models
```bash
# Train GAN
python scripts/train_gan.py --config experiments/configs/gan_config.yaml

# Train RL Agent
python scripts/train_rl.py --config experiments/configs/rl_config.yaml
```

### Launch Web Interface
```bash
python scripts/run_web_interface.py
```

## 📊 Project Structure
See `docs/project_structure.md` for detailed organization.

## 🤝 Contributing
Please read `CONTRIBUTING.md` for details on our code of conduct and submission process.

## 📜 License
This project is licensed under the MIT License - see `LICENSE` file for details.

## 🙏 Acknowledgments
- Cityscapes Dataset
- KITTI Vision Benchmark Suite
- PyTorch and Stable-Baselines3 communities
"""
        
        readme_file = self.project_root / "README.md"
        readme_file.write_text(readme_content)
        print("✅ README.md created!")
    
    def run_setup(self, install=False, use_conda=True):
        """Run complete setup process"""
        print("🚀 SafeScene AI - Environment Setup")
        print("=" * 50)
        
        # Create directory structure
        self.create_directory_structure()
        
        # Create configuration files
        self.create_requirements_file()
        self.create_conda_environment()
        self.create_gitignore()
        self.create_env_example()
        self.create_readme()
        
        # Setup git
        self.setup_git_repository()
        
        # Install dependencies if requested
        if install:
            self.install_dependencies(use_conda)
        
        print("\n✨ Setup complete!")
        print("\n📋 Next steps:")
        print("1. Activate environment: conda activate safescene-ai")
        print("2. Download datasets: python scripts/download_datasets.py")
        print("3. Start development: jupyter notebook experiments/notebooks/")

def main():
    parser = argparse.ArgumentParser(description="SafeScene AI Environment Setup")
    parser.add_argument("--project-root", type=str, default=".", 
                       help="Project root directory")
    parser.add_argument("--install", action="store_true",
                       help="Install dependencies after setup")
    parser.add_argument("--no-conda", action="store_true",
                       help="Use pip instead of conda")
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup(args.project_root)
    setup.run_setup(install=args.install, use_conda=not args.no_conda)

if __name__ == "__main__":
    main()