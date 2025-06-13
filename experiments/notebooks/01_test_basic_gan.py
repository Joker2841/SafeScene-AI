"""
SafeScene AI - Test Basic GAN Training
Quick test script to verify Phase 1 components are working.
File: experiments/notebooks/01_test_basic_gan.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our modules
from src.core.gan.generator import ConditionalStyleGAN2Generator
from src.core.gan.discriminator import StyleGAN2Discriminator
from src.data_processing.dataset_loader import DrivingSceneDataset
from src.utils.config import get_config

def test_gan_components():
    """Test if GAN components are working correctly"""
    print("🧪 Testing GAN Components...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test Generator
    print("\n1. Testing Generator...")
    generator = ConditionalStyleGAN2Generator(
        latent_dim=512,
        style_dim=512,
        n_layers=8,
        condition_dim=64
    ).to(device)
    
    # Test forward pass
    batch_size = 2
    z = torch.randn(batch_size, 512).to(device)
    conditions = generator.generate_random_conditions(batch_size, device)
    
    with torch.no_grad():
        fake_images = generator(z, **conditions)
    
    print(f"✅ Generator output shape: {fake_images.shape}")
    print(f"   Expected: [{batch_size}, 3, 512, 1024]")
    print(f"   Output range: [{fake_images.min():.2f}, {fake_images.max():.2f}]")
    
    # Test Discriminator
    print("\n2. Testing Discriminator...")
    discriminator = StyleGAN2Discriminator(
        image_size=512,
        image_width=1024,
        condition_dim=32
    ).to(device)
    
    # Create condition vector for discriminator
    disc_conditions = torch.cat([
        conditions['weather'],
        conditions['lighting'],
        conditions['layout']
    ], dim=1)
    
    with torch.no_grad():
        fake_scores = discriminator(fake_images, disc_conditions)
    
    print(f"✅ Discriminator output shape: {fake_scores.shape}")
    print(f"   Expected: [{batch_size}, 1]")
    print(f"   Score range: [{fake_scores.min():.2f}, {fake_scores.max():.2f}]")
    
    return generator, discriminator

def test_data_loading(config):
    """Test data loading pipeline"""
    print("\n🧪 Testing Data Loading...")
    
    # Create dataset
    dataset = DrivingSceneDataset(
        data_root=config.data.data_root,
        split='train',
        datasets=['cityscapes', 'kitti'],
        image_size=(config.data.image_height, config.data.image_width),
        augment=False,
        normalize=True
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("⚠️  No data found. Creating dummy data for testing...")
        return create_dummy_dataloader(config)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=DrivingSceneDataset.collate_fn,
        drop_last=True
    )
    
    # Test one batch
    for batch in dataloader:
        print(f"✅ Batch loaded successfully:")
        if isinstance(batch, dict):
            if 'image' in batch:
                print(f"   Images: {batch['image'].shape}")
            elif 'images' in batch:
                print(f"   Images: {batch['images'].shape}")
            if 'conditions' in batch:
                print(f"   Conditions: {list(batch['conditions'].keys())}")
        else:
            print(f"   Batch type: {type(batch)}")
        break
    
    return dataloader

def create_dummy_dataloader(config):
    """Create dummy dataloader for testing without real data"""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100, image_size=(512, 1024)):
            self.size = size
            self.image_size = image_size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Random image
            image = torch.randn(3, *self.image_size)
            
            # Random conditions
            weather = torch.randn(8)
            lighting = torch.randn(8)
            layout = torch.randn(16)
            
            return {
                'images': image,
                'conditions': {
                    'weather': weather,
                    'lighting': lighting,
                    'layout': layout
                }
            }
    
    dataset = DummyDataset(size=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    return dataloader

def train_one_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, device):
    """Train for one epoch"""
    generator.train()
    discriminator.train()
    
    g_losses = []
    d_losses = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Handle data format
        if isinstance(batch, dict):
            # From real dataset
            real_images = batch['image'].to(device) if 'image' in batch else batch['images'].to(device)
            
            # Get conditions if available
            if 'conditions' in batch:
                conditions = {k: v.to(device) for k, v in batch['conditions'].items()}
            else:
                # Generate random conditions
                conditions = generator.generate_random_conditions(real_images.size(0), device)
        else:
            # From dummy dataset
            real_images = batch['images'].to(device)
            conditions = {k: v.to(device) for k, v in batch['conditions'].items()}
        
        batch_size = real_images.size(0)
        
        # Prepare condition vector for discriminator
        disc_conditions = torch.cat([
            conditions['weather'],
            conditions['lighting'],
            conditions['layout']
        ], dim=1)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        
        # Real images
        real_scores = discriminator(real_images, disc_conditions)
        real_labels = torch.ones_like(real_scores)
        d_loss_real = nn.functional.binary_cross_entropy_with_logits(real_scores, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, 512).to(device)
        with torch.no_grad():
            fake_images = generator(z, **conditions)
        
        fake_scores = discriminator(fake_images, disc_conditions)
        fake_labels = torch.zeros_like(fake_scores)
        d_loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_scores, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, 512).to(device)
        fake_images = generator(z, **conditions)
        fake_scores = discriminator(fake_images, disc_conditions)
        real_labels = torch.ones_like(fake_scores)
        
        g_loss = nn.functional.binary_cross_entropy_with_logits(fake_scores, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        # Record losses
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        # Update progress bar
        progress_bar.set_postfix({
            'G_loss': f"{g_loss.item():.4f}",
            'D_loss': f"{d_loss.item():.4f}"
        })
        
        # Break after a few iterations for testing
        if batch_idx >= 10:
            break
    
    return np.mean(g_losses), np.mean(d_losses)

def visualize_samples(generator, device, save_path="test_samples.png"):
    """Generate and visualize sample images"""
    generator.eval()
    
    with torch.no_grad():
        # Generate samples with different conditions
        batch_size = 4
        z = torch.randn(batch_size, 512).to(device)
        
        # Vary conditions
        conditions_list = []
        
        # Clear day
        conditions_list.append({
            'weather': torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]).to(device),
            'lighting': torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]).to(device),
            'layout': torch.randn(1, 16).to(device)
        })
        
        # Rainy night
        conditions_list.append({
            'weather': torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]]).to(device),
            'lighting': torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]]).to(device),
            'layout': torch.randn(1, 16).to(device)
        })
        
        # Foggy dawn
        conditions_list.append({
            'weather': torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0]]).to(device),
            'lighting': torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0]]).to(device),
            'layout': torch.randn(1, 16).to(device)
        })
        
        # Random
        conditions_list.append({
            'weather': torch.randn(1, 8).to(device),
            'lighting': torch.randn(1, 8).to(device),
            'layout': torch.randn(1, 16).to(device)
        })
        
        # Generate images
        images = []
        for conditions in conditions_list:
            z_single = torch.randn(1, 512).to(device)
            image = generator(z_single, **conditions)
            images.append(image)
        
        # Combine images
        images = torch.cat(images, dim=0)
        
        # Convert to numpy and denormalize
        images = images.cpu().numpy()
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
        images = np.clip(images, 0, 1)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        titles = ['Clear Day', 'Rainy Night', 'Foggy Dawn', 'Random']
        
        for i, (ax, title) in enumerate(zip(axes.flat, titles)):
            img = images[i].transpose(1, 2, 0)
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Sample images saved to {save_path}")

def main():
    """Main testing function"""
    print("🚀 SafeScene AI - Phase 1 Component Test")
    print("=" * 50)
    
    # Load configuration
    config = get_config()
    
    # Test components
    generator, discriminator = test_gan_components()
    
    # Test data loading
    dataloader = test_data_loading(config)
    
    # Setup training
    print("\n🧪 Testing Basic Training Loop...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.0, 0.99))
    
    # Train for one epoch
    g_loss, d_loss = train_one_epoch(
        generator, discriminator, dataloader, 
        g_optimizer, d_optimizer, device
    )
    
    print(f"✅ Training completed:")
    print(f"   Average G loss: {g_loss:.4f}")
    print(f"   Average D loss: {d_loss:.4f}")
    
    # Generate samples
    print("\n🧪 Generating Sample Images...")
    save_dir = Path("experiments/results/generated_samples")
    save_dir.mkdir(parents=True, exist_ok=True)
    visualize_samples(generator, device, save_dir / "phase1_test_samples.png")
    
    print("\n✨ Phase 1 Component Test Complete!")
    print("\nNext steps:")
    print("1. Check generated samples in experiments/results/generated_samples/")
    print("2. If components work, proceed with full training")
    print("3. Implement evaluation metrics")

if __name__ == "__main__":
    main()