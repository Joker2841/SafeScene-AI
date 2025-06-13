"""
SafeScene AI - KITTI Dataset Loader
Specialized loader for KITTI dataset with proper formatting for our GAN.
File: src/data_processing/kitti_dataset.py
"""

import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
from dataclasses import dataclass
import collections

# KITTI object labels
KITTI_OBJECT_LABELS = [
    "Car", "Van", "Truck", "Pedestrian", "Person_sitting", 
    "Cyclist", "Tram", "Misc", "DontCare"
]

KITTI_OCCLUDED_LABELS = [
    "fully visible", "partly occluded", "largely occluded", "unknown"
]

@dataclass
class KittiBoundingBox:
    """KITTI bounding box representation"""
    left: float
    top: float
    right: float
    bottom: float
    
    def to_normalized(self, img_width: int, img_height: int) -> Dict[str, float]:
        """Convert to normalized coordinates [0, 1]"""
        return {
            'xmin': self.left / img_width,
            'ymin': self.top / img_height,
            'xmax': self.right / img_width,
            'ymax': self.bottom / img_height
        }
    
    def to_coco_format(self, img_width: int, img_height: int) -> List[float]:
        """Convert to COCO format [x, y, width, height]"""
        return [
            self.left,
            self.top,
            self.right - self.left,
            self.bottom - self.top
        ]

@dataclass
class KittiObject:
    """Single KITTI object annotation"""
    type: str
    truncated: float
    occluded: int
    alpha: float
    bbox: KittiBoundingBox
    dimensions: List[float]  # height, width, length in meters
    location: List[float]    # x, y, z in camera coordinates
    rotation_y: float
    
    @classmethod
    def from_kitti_line(cls, line: str):
        """Parse KITTI annotation line"""
        parts = line.strip().split(' ')
        
        return cls(
            type=parts[0],
            truncated=float(parts[1]),
            occluded=int(parts[2]),
            alpha=float(parts[3]),
            bbox=KittiBoundingBox(
                left=float(parts[4]),
                top=float(parts[5]),
                right=float(parts[6]),
                bottom=float(parts[7])
            ),
            dimensions=[float(parts[8]), float(parts[9]), float(parts[10])],
            location=[float(parts[11]), float(parts[12]), float(parts[13])],
            rotation_y=float(parts[14])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'type': self.type,
            'truncated': self.truncated,
            'occluded': self.occluded,
            'alpha': self.alpha,
            'bbox': {
                'left': self.bbox.left,
                'top': self.bbox.top,
                'right': self.bbox.right,
                'bottom': self.bbox.bottom
            },
            'dimensions': self.dimensions,
            'location': self.location,
            'rotation_y': self.rotation_y
        }

class KittiDataset(Dataset):
    """
    KITTI dataset loader for SafeScene AI
    
    Directory structure expected:
    kitti/
    ├── training/
    │   ├── image_2/         # Left color camera images
    │   ├── label_2/         # Object labels
    │   ├── calib/           # Calibration files
    │   └── velodyne/        # Optional: LiDAR data
    └── testing/
        └── image_2/
    """
    
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 transform=None,
                 target_size: Tuple[int, int] = (512, 1024),
                 use_depth: bool = False,
                 use_lidar: bool = False,
                 filter_classes: Optional[List[str]] = None):
        """
        Args:
            root_dir: Path to KITTI dataset root
            split: 'train', 'val', or 'test'
            transform: Optional image transformations
            target_size: Target image size (height, width)
            use_depth: Whether to load depth maps
            use_lidar: Whether to load LiDAR data
            filter_classes: List of object classes to include (None = all)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.use_depth = use_depth
        self.use_lidar = use_lidar
        self.filter_classes = filter_classes
        
        # Setup paths
        self.images_dir = self.root_dir / 'training' / 'image_2'
        self.labels_dir = self.root_dir / 'training' / 'label_2'
        self.calib_dir = self.root_dir / 'training' / 'calib'
        self.velodyne_dir = self.root_dir / 'training' / 'velodyne'
        
        # Get file list
        self.image_files = sorted(list(self.images_dir.glob('*.png')))
        
        # Split dataset (if no official split available)
        self._create_splits()
        
        print(f"Loaded KITTI dataset: {len(self.image_files)} images for {split}")
    
    def _create_splits(self):
        """Create train/val/test splits"""
        # KITTI doesn't have official train/val split, so we create one
        # Using 80/10/10 split
        np.random.seed(42)
        indices = np.arange(len(self.image_files))
        np.random.shuffle(indices)
        
        train_size = int(0.8 * len(indices))
        val_size = int(0.1 * len(indices))
        
        if self.split == 'train':
            indices = indices[:train_size]
        elif self.split == 'val':
            indices = indices[train_size:train_size + val_size]
        else:  # test
            indices = indices[train_size + val_size:]
        
        self.image_files = [self.image_files[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get single sample"""
        image_path = self.image_files[idx]
        image_id = image_path.stem
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Resize image
        image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        
        # Load annotations
        label_path = self.labels_dir / f"{image_id}.txt"
        objects = self._load_annotations(label_path, orig_width, orig_height)
        
        # Filter objects by class if specified
        if self.filter_classes:
            objects = [obj for obj in objects if obj.type in self.filter_classes]
        
        # Load calibration
        calib_path = self.calib_dir / f"{image_id}.txt"
        calibration = self._load_calibration(calib_path) if calib_path.exists() else None
        
        # Create scene parameters for GAN conditioning
        scene_params = self._extract_scene_parameters(objects, calibration)
        
        # Convert to tensors
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
        
        # Create condition vectors
        conditions = self._create_condition_vectors(scene_params)
        
        sample = {
            'image': image,
            'image_id': image_id,
            'objects': [obj.to_dict() for obj in objects],
            'scene_params': scene_params,
            'conditions': conditions,
            'original_size': (orig_width, orig_height)
        }
        
        # Add optional data
        if self.use_lidar:
            lidar_path = self.velodyne_dir / f"{image_id}.bin"
            if lidar_path.exists():
                sample['lidar'] = self._load_lidar(lidar_path)
        
        if calibration:
            sample['calibration'] = calibration
        
        return sample
    
    def _load_annotations(self, label_path: Path, img_width: int, img_height: int) -> List[KittiObject]:
        """Load KITTI annotations from label file"""
        objects = []
        
        if not label_path.exists():
            return objects
        
        with open(label_path, 'r') as f:
            for line in f:
                obj = KittiObject.from_kitti_line(line)
                if obj.type != 'DontCare':
                    objects.append(obj)
        
        return objects
    
    def _load_calibration(self, calib_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """Load calibration matrices"""
        if not calib_path.exists():
            return None
        
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key] = np.array([float(x) for x in value.strip().split()])
        
        # Reshape matrices
        if 'P0' in calib:
            calib['P0'] = calib['P0'].reshape(3, 4)
        if 'P1' in calib:
            calib['P1'] = calib['P1'].reshape(3, 4)
        if 'P2' in calib:
            calib['P2'] = calib['P2'].reshape(3, 4)
        if 'P3' in calib:
            calib['P3'] = calib['P3'].reshape(3, 4)
        if 'R0_rect' in calib:
            calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
        if 'Tr_velo_to_cam' in calib:
            calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
        
        return calib
    
    def _load_lidar(self, lidar_path: Path) -> np.ndarray:
        """Load LiDAR point cloud"""
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
        return points  # x, y, z, reflectance
    
    def _extract_scene_parameters(self, objects: List[KittiObject], 
                                 calibration: Optional[Dict]) -> Dict[str, Any]:
        """Extract scene parameters for GAN conditioning"""
        # Count objects by type
        object_counts = collections.Counter(obj.type for obj in objects)
        
        # Estimate scene complexity
        num_vehicles = sum(object_counts.get(cls, 0) for cls in ['Car', 'Van', 'Truck'])
        num_pedestrians = object_counts.get('Pedestrian', 0) + object_counts.get('Person_sitting', 0)
        num_cyclists = object_counts.get('Cyclist', 0)
        
        # Estimate occlusion level
        occlusion_levels = [obj.occluded for obj in objects]
        avg_occlusion = np.mean(occlusion_levels) if occlusion_levels else 0
        
        # Scene density (objects per image area)
        total_objects = len(objects)
        
        # Estimate scene type based on object composition
        if num_vehicles > 5:
            scene_type = 'highway'
        elif num_pedestrians > 3:
            scene_type = 'urban'
        else:
            scene_type = 'suburban'
        
        return {
            'num_vehicles': num_vehicles,
            'num_pedestrians': num_pedestrians,
            'num_cyclists': num_cyclists,
            'total_objects': total_objects,
            'avg_occlusion': avg_occlusion,
            'scene_type': scene_type,
            'object_density': total_objects / 20.0,  # Normalized
            'weather': 'clear',  # KITTI is mostly clear weather
            'time_of_day': 'day',  # KITTI is daytime
        }
    
    def _create_condition_vectors(self, scene_params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Create condition vectors for GAN"""
        # Weather vector (KITTI is mostly clear weather)
        weather = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        
        # Lighting vector (KITTI is daytime)
        lighting = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        
        # Scene layout vector
        layout = torch.zeros(16, dtype=torch.float32)
        layout[0] = scene_params['num_vehicles'] / 10.0
        layout[1] = scene_params['num_pedestrians'] / 5.0
        layout[2] = scene_params['num_cyclists'] / 3.0
        layout[3] = scene_params['object_density']
        layout[4] = scene_params['avg_occlusion'] / 3.0
        
        # Scene type encoding
        scene_type_map = {'urban': 5, 'suburban': 6, 'highway': 7}
        if scene_params['scene_type'] in scene_type_map:
            layout[scene_type_map[scene_params['scene_type']]] = 1.0
        
        return {
            'weather': weather,
            'lighting': lighting,
            'layout': layout
        }
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for batching"""
        # Stack images
        images = torch.stack([item['image'] for item in batch])
        
        # Stack conditions
        conditions = {
            'weather': torch.stack([item['conditions']['weather'] for item in batch]),
            'lighting': torch.stack([item['conditions']['lighting'] for item in batch]),
            'layout': torch.stack([item['conditions']['layout'] for item in batch])
        }
        
        # Collect other data
        collated = {
            'images': images,
            'conditions': conditions,
            'image_ids': [item['image_id'] for item in batch],
            'objects': [item['objects'] for item in batch],
            'scene_params': [item['scene_params'] for item in batch]
        }
        
        return collated

def create_kitti_dataloader(root_dir: str, 
                           batch_size: int = 32,
                           split: str = 'train',
                           num_workers: int = 4,
                           **kwargs) -> torch.utils.data.DataLoader:
    """Create KITTI dataloader with proper settings"""
    
    dataset = KittiDataset(
        root_dir=root_dir,
        split=split,
        **kwargs
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader

if __name__ == "__main__":
    # Test KITTI dataset loading
    dataset = KittiDataset(
        root_dir="./data/raw/kitti",
        split='train',
        target_size=(512, 1024),
        filter_classes=['Car', 'Pedestrian', 'Cyclist']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test single sample
        sample = dataset[0]
        print(f"\nSample data:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Number of objects: {len(sample['objects'])}")
        print(f"  Scene parameters: {sample['scene_params']}")
        print(f"  Conditions shapes:")
        for k, v in sample['conditions'].items():
            print(f"    {k}: {v.shape}")
        
        # Test dataloader
        dataloader = create_kitti_dataloader(
            root_dir="./data/raw/kitti",
            batch_size=4,
            split='train'
        )
        
        for batch in dataloader:
            print(f"\nBatch data:")
            print(f"  Images shape: {batch['images'].shape}")
            print(f"  Weather conditions: {batch['conditions']['weather'].shape}")
            print(f"  Layout conditions: {batch['conditions']['layout'].shape}")
            break