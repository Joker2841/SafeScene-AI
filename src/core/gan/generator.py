"""
SafeScene AI - Conditional StyleGAN2 Generator
Generates synthetic driving scenes with conditional controls.
File: src/core/gan/generator.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import math

class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 lr_multiplier: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        
        self.lr_multiplier = lr_multiplier
        self.scale = (1 / math.sqrt(in_features)) * lr_multiplier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            out = F.linear(x, self.weight * self.scale, self.bias * self.lr_multiplier)
        else:
            out = F.linear(x, self.weight * self.scale)
        return out

class ConstantInput(nn.Module):
    """Learned constant input"""
    def __init__(self, channel: int, size: Tuple[int, int] = (4, 8)):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, channel, size[0], size[1]))
    
    def forward(self, batch_size: int):
        return self.const.repeat(batch_size, 1, 1, 1)

class StyledConv(nn.Module):
    """Styled convolution layer"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 style_dim: int, upsample: bool = False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        
        if upsample:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, 
                stride=2, padding=kernel_size//2, output_padding=1
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=1, padding=kernel_size//2
            )
        
        self.modulation = EqualLinear(style_dim, in_channels)
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Apply style modulation
        style = self.modulation(style).view(-1, self.in_channels, 1, 1)
        
        # Modulate weights
        x = x * style
        
        # Apply convolution
        out = self.conv(x) * self.scale
        
        return out

class NoiseInjection(nn.Module):
    """Inject noise for stochastic variation"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
    
    def forward(self, image: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            batch, _, height, width = image.shape
            noise = torch.randn(batch, 1, height, width, device=image.device)
        
        return image + self.weight * noise

class StyledConvBlock(nn.Module):
    """StyleGAN2 synthesis block"""
    def __init__(self, in_channels: int, out_channels: int, style_dim: int,
                 upsample: bool = False):
        super().__init__()
        
        self.conv1 = StyledConv(in_channels, out_channels, 3, style_dim, upsample)
        self.noise1 = NoiseInjection()
        self.activate1 = nn.LeakyReLU(0.2)
        
        self.conv2 = StyledConv(out_channels, out_channels, 3, style_dim)
        self.noise2 = NoiseInjection()
        self.activate2 = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor, 
                noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None)) -> torch.Tensor:
        out = self.conv1(x, style)
        out = self.noise1(out, noise[0])
        out = self.activate1(out)
        
        out = self.conv2(out, style)
        out = self.noise2(out, noise[1])
        out = self.activate2(out)
        
        return out

class ToRGB(nn.Module):
    """Convert feature maps to RGB image"""
    def __init__(self, in_channels: int, style_dim: int):
        super().__init__()
        self.conv = StyledConv(in_channels, 3, 1, style_dim)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, style)
        out = out + self.bias
        return out

class MappingNetwork(nn.Module):
    """Map latent codes to style codes with conditional inputs"""
    def __init__(self, latent_dim: int, style_dim: int, num_layers: int = 8,
                 condition_dim: int = 0):
        super().__init__()
        
        layers = []
        in_dim = latent_dim + condition_dim
        
        for i in range(num_layers):
            layers.append(EqualLinear(in_dim, style_dim, lr_multiplier=0.01))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = style_dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        if condition is not None:
            x = torch.cat([x, condition], dim=1)
        return self.net(x)

class ConditionalStyleGAN2Generator(nn.Module):
    """Main generator with conditional controls for driving scenes"""
    def __init__(self, latent_dim: int = 512, style_dim: int = 512,
                 n_layers: int = 8, condition_dim: int = 64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.n_layers = n_layers
        
        # Mapping network with conditional input
        self.mapping = MappingNetwork(latent_dim, style_dim, 8, condition_dim)
        
        # Starting constant
        self.const = ConstantInput(512, (4, 8))  # 4x8 for 512x1024 output
        
        # Synthesis network
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        
        # Channel progression
        channels = [512, 512, 512, 512, 256, 128, 64, 32]
        
        # First block (no upsampling)
        self.convs.append(StyledConvBlock(512, 512, style_dim, upsample=False))
        self.to_rgbs.append(ToRGB(512, style_dim))
        
        # Progressive blocks with upsampling
        # 4x8 -> 8x16 -> 16x32 -> 32x64 -> 64x128 -> 128x256 -> 256x512 -> 512x1024
        in_channel = 512
        for i in range(7):  # 7 upsampling blocks
            out_channel = channels[i+1]
            self.convs.append(StyledConvBlock(in_channel, out_channel, style_dim, upsample=True))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        
        # Condition encoder for weather, lighting, scene layout
        self.condition_encoder = nn.Sequential(
            nn.Linear(32, 64),  # Input: weather(8) + lighting(8) + layout(16)
            nn.LeakyReLU(0.2),
            nn.Linear(64, condition_dim),
            nn.LeakyReLU(0.2)
        )
    
    def encode_conditions(self, weather: torch.Tensor, lighting: torch.Tensor,
                         layout: torch.Tensor) -> torch.Tensor:
        """Encode conditional inputs"""
        conditions = torch.cat([weather, lighting, layout], dim=1)
        return self.condition_encoder(conditions)
    
    def forward(self, z: torch.Tensor, 
                weather: Optional[torch.Tensor] = None,
                lighting: Optional[torch.Tensor] = None,
                layout: Optional[torch.Tensor] = None,
                truncation: float = 1.0,
                noise: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Generate driving scene image
        
        Args:
            z: Latent code [batch, latent_dim]
            weather: Weather condition encoding [batch, 8]
            lighting: Lighting condition encoding [batch, 8]
            layout: Scene layout encoding [batch, 16]
            truncation: Truncation factor for sampling
            noise: Optional noise tensors for each layer
        
        Returns:
            Generated image [batch, 3, 512, 1024]
        """
        batch_size = z.shape[0]
        
        # Encode conditions if provided
        condition = None
        if weather is not None and lighting is not None and layout is not None:
            condition = self.encode_conditions(weather, lighting, layout)
        
        # Get style codes
        w = self.mapping(z, condition)
        
        # Truncation trick
        if truncation < 1.0:
            w = truncation * w
        
        # Initial constant
        x = self.const(batch_size)
        
        # Progressive synthesis
        rgb = None
        for i, (conv, to_rgb) in enumerate(zip(self.convs, self.to_rgbs)):
            # Apply convolution
            x = conv(x, w)
            
            # Convert to RGB and accumulate
            if i == 0:
                rgb = to_rgb(x, w)
            else:
                # Upsample previous RGB to match current resolution
                rgb = F.interpolate(rgb, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
                rgb = rgb + to_rgb(x, w)
        
        # Final activation
        rgb = torch.tanh(rgb)
        
        # Ensure output is exactly 512x1024
        if rgb.shape[2:] != (512, 1024):
            rgb = F.interpolate(rgb, size=(512, 1024), mode='bilinear', align_corners=False)
        
        return rgb
    
    def generate_random_conditions(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Generate random condition vectors for testing"""
        weather = torch.randn(batch_size, 8, device=device)
        lighting = torch.randn(batch_size, 8, device=device)
        layout = torch.randn(batch_size, 16, device=device)
        
        return {
            'weather': weather,
            'lighting': lighting,
            'layout': layout
        }

if __name__ == "__main__":
    # Test generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = ConditionalStyleGAN2Generator().to(device)
    
    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, 512, device=device)
    conditions = generator.generate_random_conditions(batch_size, device)
    
    with torch.no_grad():
        images = generator(z, **conditions)
    
    print(f"Generator output shape: {images.shape}")  # Should be [4, 3, 512, 1024]
    print(f"Output range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Model parameters: {sum(p.numel() for p in generator.parameters()):,}")