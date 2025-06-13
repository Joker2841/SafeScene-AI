"""
SafeScene AI - RL Environment for Intelligent Scene Generation
Environment where RL agent learns to guide GAN generation.
File: src/core/rl/environment.py
"""

import gym
from gym import spaces
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
import cv2

@dataclass
class SceneParameters:
    """Parameters that define a driving scene"""
    # Weather conditions (0-1 normalized)
    rain_intensity: float = 0.0
    fog_density: float = 0.0
    snow_intensity: float = 0.0
    
    # Lighting conditions (0-1 normalized)
    time_of_day: float = 0.5  # 0=night, 0.5=noon, 1=night
    sun_angle: float = 0.5
    brightness: float = 0.5
    contrast: float = 0.5
    
    # Scene complexity
    num_vehicles: int = 5
    num_pedestrians: int = 2
    object_density: float = 0.3
    
    # Camera parameters
    camera_height: float = 1.5
    camera_angle: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert parameters to vector for neural network input"""
        return np.array([
            self.rain_intensity,
            self.fog_density,
            self.snow_intensity,
            self.time_of_day,
            self.sun_angle,
            self.brightness,
            self.contrast,
            self.num_vehicles / 20.0,  # Normalize
            self.num_pedestrians / 10.0,  # Normalize
            self.object_density,
            self.camera_height / 3.0,  # Normalize
            self.camera_angle / 90.0  # Normalize
        ], dtype=np.float32)
    
    def from_vector(self, vector: np.ndarray):
        """Update parameters from vector"""
        self.rain_intensity = float(vector[0])
        self.fog_density = float(vector[1])
        self.snow_intensity = float(vector[2])
        self.time_of_day = float(vector[3])
        self.sun_angle = float(vector[4])
        self.brightness = float(vector[5])
        self.contrast = float(vector[6])
        self.num_vehicles = int(vector[7] * 20)
        self.num_pedestrians = int(vector[8] * 10)
        self.object_density = float(vector[9])
        self.camera_height = float(vector[10] * 3.0)
        self.camera_angle = float(vector[11] * 90.0)

class SafeSceneEnv(gym.Env):
    """
    RL Environment for intelligent scene generation.
    The agent learns to set scene parameters that generate valuable training data.
    """
    
    def __init__(self, generator_model=None, discriminator_model=None,
                 max_steps: int = 100, device: str = 'cuda'):
        super().__init__()
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.generator = generator_model
        self.discriminator = discriminator_model
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action space (continuous control of scene parameters)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),  # Number of controllable parameters
            dtype=np.float32
        )
        
        # Define observation space
        # Includes: current parameters, quality metrics, diversity metrics
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(64,),  # Encoded state representation
            dtype=np.float32
        )
        
        # Scene generation history for diversity calculation
        self.scene_history = []
        self.generated_images = []
        
        # Current scene parameters
        self.current_params = SceneParameters()
        
        # Metrics tracking
        self.quality_scores = []
        self.diversity_scores = []
        self.difficulty_scores = []
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_params = SceneParameters()
        self.scene_history = []
        self.generated_images = []
        self.quality_scores = []
        self.diversity_scores = []
        self.difficulty_scores = []
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return results
        
        Args:
            action: Parameter adjustments [-1, 1]
        
        Returns:
            observation: New state
            reward: Reward signal
            done: Episode finished
            info: Additional information
        """
        # Update scene parameters based on action
        self._update_parameters(action)
        
        # Generate scene with current parameters
        generated_image, quality_score = self._generate_scene()
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity()
        
        # Calculate difficulty score (how challenging for AV models)
        difficulty_score = self._calculate_difficulty()
        
        # Calculate reward
        reward = self._calculate_reward(
            quality_score, diversity_score, difficulty_score
        )
        
        # Store in history
        self.scene_history.append(self.current_params.to_vector().copy())
        self.generated_images.append(generated_image)
        self.quality_scores.append(quality_score)
        self.diversity_scores.append(diversity_score)
        self.difficulty_scores.append(difficulty_score)
        
        # Update step count
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Get new observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'quality_score': quality_score,
            'diversity_score': diversity_score,
            'difficulty_score': difficulty_score,
            'scene_parameters': self.current_params,
            'generated_image': generated_image
        }
        
        return observation, reward, done, info
    
    def _update_parameters(self, action: np.ndarray):
        """Update scene parameters based on action"""
        # Get current parameter vector
        current_vector = self.current_params.to_vector()
        
        # Apply action with scaling
        action_scale = 0.1  # Limit change rate
        new_vector = current_vector + action * action_scale
        
        # Clip to valid ranges
        new_vector = np.clip(new_vector, 0.0, 1.0)
        
        # Update parameters
        self.current_params.from_vector(new_vector)
    
    def _generate_scene(self) -> Tuple[Optional[np.ndarray], float]:
        """Generate scene using current parameters"""
        if self.generator is None:
            # Return dummy data for testing
            dummy_image = np.random.rand(512, 1024, 3)
            quality_score = np.random.rand()
            return dummy_image, quality_score
        
        # Convert parameters to condition vectors
        weather = torch.tensor([
            self.current_params.rain_intensity,
            self.current_params.fog_density,
            self.current_params.snow_intensity,
            0.0, 0.0, 0.0, 0.0, 0.0  # Padding to 8 dims
        ]).unsqueeze(0).to(self.device)
        
        lighting = torch.tensor([
            self.current_params.time_of_day,
            self.current_params.sun_angle,
            self.current_params.brightness,
            self.current_params.contrast,
            0.0, 0.0, 0.0, 0.0  # Padding to 8 dims
        ]).unsqueeze(0).to(self.device)
        
        layout = torch.tensor([
            self.current_params.num_vehicles / 20.0,
            self.current_params.num_pedestrians / 10.0,
            self.current_params.object_density,
            self.current_params.camera_height / 3.0,
            self.current_params.camera_angle / 90.0,
            *([0.0] * 11)  # Padding to 16 dims
        ]).unsqueeze(0).to(self.device)
        
        # Generate latent code
        z = torch.randn(1, 512).to(self.device)
        
        # Generate image
        with torch.no_grad():
            generated = self.generator(z, weather, lighting, layout)
            
            # Calculate quality score using discriminator
            if self.discriminator is not None:
                conditions = torch.cat([weather, lighting, layout], dim=1)
                quality_score = torch.sigmoid(
                    self.discriminator(generated, conditions)
                ).item()
            else:
                quality_score = 0.5
        
        # Convert to numpy
        generated_np = generated.squeeze(0).permute(1, 2, 0).cpu().numpy()
        generated_np = ((generated_np + 1) * 127.5).astype(np.uint8)
        
        return generated_np, quality_score
    
    def _calculate_diversity(self) -> float:
        """Calculate diversity score based on parameter history"""
        if len(self.scene_history) < 2:
            return 1.0
        
        # Calculate average distance to recent scenes
        current_vector = self.current_params.to_vector()
        recent_scenes = self.scene_history[-10:]  # Last 10 scenes
        
        distances = []
        for past_scene in recent_scenes:
            distance = np.linalg.norm(current_vector - past_scene)
            distances.append(distance)
        
        # Normalize diversity score
        avg_distance = np.mean(distances)
        diversity_score = min(avg_distance / 2.0, 1.0)  # Normalize to [0, 1]
        
        return diversity_score
    
    def _calculate_difficulty(self) -> float:
        """Calculate difficulty score for autonomous vehicle models"""
        # Higher difficulty for:
        # - Extreme weather conditions
        # - Low visibility (night, fog)
        # - High object density
        # - Unusual camera angles
        
        weather_difficulty = max(
            self.current_params.rain_intensity,
            self.current_params.fog_density,
            self.current_params.snow_intensity
        )
        
        lighting_difficulty = abs(self.current_params.time_of_day - 0.5) * 2  # Night/dawn/dusk
        
        complexity_difficulty = (
            self.current_params.object_density * 0.5 +
            min(self.current_params.num_vehicles / 20.0, 1.0) * 0.3 +
            min(self.current_params.num_pedestrians / 10.0, 1.0) * 0.2
        )
        
        camera_difficulty = abs(self.current_params.camera_angle) / 90.0
        
        # Combine difficulties
        difficulty_score = (
            weather_difficulty * 0.4 +
            lighting_difficulty * 0.3 +
            complexity_difficulty * 0.2 +
            camera_difficulty * 0.1
        )
        
        return min(difficulty_score, 1.0)
    
    def _calculate_reward(self, quality: float, diversity: float, difficulty: float) -> float:
        """
        Calculate reward based on multiple objectives
        
        Good scenes should be:
        - High quality (realistic)
        - Diverse (different from recent generations)
        - Appropriately difficult (challenging but not impossible)
        """
        # Quality component (want high quality)
        quality_reward = quality
        
        # Diversity component (want high diversity)
        diversity_reward = diversity
        
        # Difficulty component (want moderate to high difficulty)
        # Penalize too easy or impossibly hard scenes
        if difficulty < 0.3:
            difficulty_reward = difficulty / 0.3 * 0.5
        elif difficulty > 0.8:
            difficulty_reward = 0.5 + (1 - difficulty) / 0.2 * 0.5
        else:
            difficulty_reward = 1.0
        
        # Combine rewards
        total_reward = (
            quality_reward * 0.3 +
            diversity_reward * 0.3 +
            difficulty_reward * 0.4
        )
        
        # Add bonus for balanced scenes
        balance_bonus = 0.0
        if quality > 0.7 and diversity > 0.5 and 0.4 < difficulty < 0.7:
            balance_bonus = 0.2
        
        return total_reward + balance_bonus
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for RL agent"""
        # Include current parameters
        param_vector = self.current_params.to_vector()
        
        # Include recent history statistics
        if len(self.quality_scores) > 0:
            recent_quality = np.mean(self.quality_scores[-5:])
            recent_diversity = np.mean(self.diversity_scores[-5:])
            recent_difficulty = np.mean(self.difficulty_scores[-5:])
        else:
            recent_quality = 0.5
            recent_diversity = 0.5
            recent_difficulty = 0.5
        
        # Include global statistics
        if len(self.scene_history) > 0:
            param_variance = np.var(self.scene_history[-20:], axis=0).mean()
        else:
            param_variance = 0.0
        
        # Progress indicator
        progress = self.current_step / self.max_steps
        
        # Combine into observation
        observation = np.concatenate([
            param_vector,  # 12 dims
            [recent_quality, recent_diversity, recent_difficulty],  # 3 dims
            [param_variance, progress],  # 2 dims
            np.zeros(47)  # Padding to 64 dims
        ])
        
        return observation.astype(np.float32)
    
    def render(self, mode='human'):
        """Render current state"""
        if len(self.generated_images) > 0:
            latest_image = self.generated_images[-1]
            
            if mode == 'human':
                cv2.imshow('Generated Scene', cv2.cvtColor(latest_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            elif mode == 'rgb_array':
                return latest_image
    
    def close(self):
        """Clean up resources"""
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test environment
    env = SafeSceneEnv(generator_model=None, discriminator_model=None)
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test step
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Quality: {info['quality_score']:.3f}")
        print(f"  Diversity: {info['diversity_score']:.3f}")
        print(f"  Difficulty: {info['difficulty_score']:.3f}")
        print(f"  Done: {done}")
    
    env.close()