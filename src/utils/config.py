"""
SafeScene AI - Configuration Management System
Central configuration management for all components.
File: src/utils/config.py
"""

import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DataConfig:
    """Data-related configuration"""
    data_root: str = "./data"
    raw_data_dir: str = "raw"
    processed_data_dir: str = "processed"
    synthetic_data_dir: str = "synthetic"
    
    # Dataset specific
    cityscapes_root: str = "cityscapes"
    kitti_root: str = "kitti"
    custom_edge_cases_root: str = "custom_edge_cases"
    
    # Image settings
    image_height: int = 512
    image_width: int = 1024
    image_channels: int = 3
    
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    def __post_init__(self):
        """Create full paths"""
        self.data_root = Path(self.data_root)
        self.raw_path = self.data_root / self.raw_data_dir
        self.processed_path = self.data_root / self.processed_data_dir
        self.synthetic_path = self.data_root / self.synthetic_data_dir

@dataclass
class GANConfig:
    """GAN model configuration"""
    # Architecture
    model_type: str = "conditional_stylegan2"
    latent_dim: int = 512
    style_dim: int = 512
    n_layers: int = 8
    
    # Generator
    g_learning_rate: float = 0.002
    g_beta1: float = 0.0
    g_beta2: float = 0.99
    g_reg_interval: int = 4
    
    # Discriminator
    d_learning_rate: float = 0.002
    d_beta1: float = 0.0
    d_beta2: float = 0.99
    d_reg_interval: int = 16
    
    # Training
    batch_size: int = 16
    gradient_accumulation: int = 2
    mixed_precision: bool = True
    
    # Loss weights
    adversarial_weight: float = 1.0
    perceptual_weight: float = 1.0
    l1_weight: float = 10.0
    
    # Conditional controls
    use_weather_condition: bool = True
    use_lighting_condition: bool = True
    use_scene_layout: bool = True
    
    # Checkpointing
    checkpoint_interval: int = 1000
    save_sample_interval: int = 500

@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    # Environment
    env_name: str = "SafeSceneEnv-v0"
    observation_space_dim: int = 256
    action_space_dim: int = 64
    max_episode_steps: int = 100
    
    # Agent
    algorithm: str = "PPO"
    policy_type: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Reward design
    realism_reward_weight: float = 0.3
    diversity_reward_weight: float = 0.3
    difficulty_reward_weight: float = 0.4
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 5
    difficulty_increase_rate: float = 0.1
    
    # Training
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    save_freq: int = 50_000
    
    # Exploration
    ent_coef: float = 0.01
    clip_range: float = 0.2

@dataclass
class TrainingConfig:
    """General training configuration"""
    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Paths
    checkpoint_dir: str = "./experiments/results/checkpoints"
    log_dir: str = "./experiments/results/logs"
    sample_dir: str = "./experiments/results/generated_samples"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = os.environ.get("WANDB_PROJECT", "safescene-ai")
    wandb_entity: str = os.environ.get("WANDB_ENTITY", None)
    use_tensorboard: bool = True
    log_interval: int = 100
    
    # Training schedule
    num_epochs: int = 100
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    
    # Optimization
    gradient_clip_val: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    
    def __post_init__(self):
        """Create directories"""
        for path in [self.checkpoint_dir, self.log_dir, self.sample_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)

@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration"""
    # Metrics to compute
    compute_fid: bool = True
    compute_kid: bool = True
    compute_lpips: bool = True
    compute_ssim: bool = True
    compute_is: bool = True
    
    # FID/KID settings
    fid_batch_size: int = 64
    fid_num_samples: int = 10000
    kid_subset_size: int = 1000
    
    # Evaluation frequency
    eval_interval: int = 5000
    full_eval_interval: int = 10000
    
    # Thresholds for success
    target_fid: float = 40.0
    target_kid: float = 0.05
    target_lpips: float = 0.3
    
    # Diversity metrics
    compute_diversity: bool = True
    diversity_k: int = 5
    
    # Safety metrics
    compute_safety_score: bool = True
    edge_case_coverage: float = 0.95

@dataclass
class WebInterfaceConfig:
    """Web interface configuration"""
    # API settings
    api_host: str = os.environ.get("API_HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("API_PORT", 8000))
    
    # Frontend settings
    enable_live_generation: bool = True
    max_concurrent_generations: int = 3
    generation_timeout: int = 60
    
    # Display settings
    preview_size: tuple = (512, 256)
    gallery_size: int = 20
    
    # Cache settings
    cache_generated_images: bool = True
    cache_size_mb: int = 1000
    
    # Security
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 30

class Config:
    """Main configuration class combining all configs"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.data = DataConfig()
        self.gan = GANConfig()
        self.rl = RLConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.web = WebInterfaceConfig()
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations
        for key, value in config_dict.items():
            if hasattr(self, key):
                config_obj = getattr(self, key)
                for field_name, field_value in value.items():
                    if hasattr(config_obj, field_name):
                        setattr(config_obj, field_name, field_value)
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'gan': self.gan.__dict__,
            'rl': self.rl.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'web': self.web.__dict__
        }
        
        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            return obj
        
        config_dict = convert_paths(config_dict)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'data': self.data.__dict__,
            'gan': self.gan.__dict__,
            'rl': self.rl.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'web': self.web.__dict__
        }
    
    def update_from_args(self, args: Dict[str, Any]):
        """Update configuration from command line arguments"""
        for key, value in args.items():
            if '.' in key:
                # Handle nested attributes like 'gan.batch_size'
                parts = key.split('.')
                config_section = getattr(self, parts[0])
                if hasattr(config_section, parts[1]):
                    setattr(config_section, parts[1], value)
            else:
                # Handle top-level attributes
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check data paths exist
        if not self.data.data_root.exists():
            issues.append(f"Data root directory {self.data.data_root} does not exist")
        
        # Check batch sizes
        if self.gan.batch_size <= 0:
            issues.append("GAN batch size must be positive")
        
        # Check device
        if self.training.device == "cuda" and not torch.cuda.is_available():
            issues.append("CUDA device requested but not available")
        
        # Check value ranges
        if not 0 <= self.rl.gamma <= 1:
            issues.append("RL gamma must be between 0 and 1")
        
        return issues

# Singleton instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration singleton"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance

def reset_config():
    """Reset configuration singleton"""
    global _config_instance
    _config_instance = None

# Import guard for torch
try:
    import torch
except ImportError:
    torch = None

if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Device: {config.training.device}")
    print(f"Data root: {config.data.data_root}")
    
    # Validate
    issues = config.validate()
    if issues:
        print("\nConfiguration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration is valid!")