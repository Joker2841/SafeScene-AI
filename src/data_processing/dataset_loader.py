"""
SafeScene AI - Dataset Loader
Handles loading and preprocessing of driving scene datasets.
File: src/data_processing/dataset_loader.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
from tqdm import tqdm

class DrivingSceneDataset(Dataset):
    """
    Unified dataset for loading Cityscapes, KITTI, and custom edge cases
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 datasets: List[str] = ['cityscapes', 'kitti'],
                 image_size: Tuple[int, int] = (512, 1024),
                 augment: bool = True,
                 normalize: bool = True):
        """
        Args:
            data_root: Root directory containing datasets
            split: 'train', 'val', or 'test'
            datasets: List of datasets to load
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images
        """
        self.data_root = Path(data_root)
        self.split = split
        self.datasets = datasets
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        self.normalize = normalize

        # Initialize data lists
        self.image_paths = []
        self.annotation_paths = []
        self.metadata = []

        # Load data from each dataset
        for dataset in datasets:
            self._load_dataset(dataset)

        # Create transforms
        self.transform = self._create_transforms()

        print(f"Loaded {len(self.image_paths)} images for {split} split")

    def _load_dataset(self, dataset_name: str):
        """Load specific dataset"""
        dataset_path = self.data_root / 'raw' / dataset_name

        if not dataset_path.exists():
            print(f"Warning: Dataset {dataset_name} not found at {dataset_path}")
            return

        if dataset_name == 'cityscapes':
            self._load_cityscapes(dataset_path)
        elif dataset_name == 'kitti':
            self._load_kitti(dataset_path)
        elif dataset_name == 'custom_edge_cases':
            self._load_edge_cases(dataset_path)
        else:
            print(f"Unknown dataset: {dataset_name}")

    def _load_cityscapes(self, dataset_path: Path):
        """Load Cityscapes dataset"""
        img_dir = dataset_path / 'leftImg8bit' / self.split
        ann_dir = dataset_path / 'gtFine' / self.split

        if not img_dir.exists():
            return

        # Find all images
        for city_dir in img_dir.iterdir():
            if city_dir.is_dir():
                for img_path in city_dir.glob('*_leftImg8bit.png'):
                    # Find corresponding annotation
                    ann_name = img_path.stem.replace('_leftImg8bit', '_gtFine_labelIds')
                    ann_path = ann_dir / city_dir.name / f"{ann_name}.png"

                    self.image_paths.append(str(img_path))
                    self.annotation_paths.append(str(ann_path) if ann_path.exists() else None)

                    # Extract metadata
                    self.metadata.append({
                        'dataset': 'cityscapes',
                        'city': city_dir.name,
                        'weather': 'clear',  # Cityscapes is mostly clear weather
                        'time_of_day': 'day',
                        'scene_type': 'urban'
                    })

    def _load_kitti(self, dataset_path: Path):
        """Load KITTI dataset"""
        img_dir = dataset_path / 'training' / 'image_2'
        ann_dir = dataset_path / 'training' / 'label_2'

        if not img_dir.exists():
            return

        # Find all images
        for img_path in img_dir.glob('*.png'):
            ann_path = ann_dir / img_path.stem / '.txt'

            self.image_paths.append(str(img_path))
            self.annotation_paths.append(str(ann_path) if ann_path.exists() else None)

            self.metadata.append({
                'dataset': 'kitti',
                'weather': 'clear',
                'time_of_day': 'day',
                'scene_type': 'mixed'
            })

    def _load_edge_cases(self, dataset_path: Path):
        """Load custom edge cases"""
        for category_dir in dataset_path.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('.'):
                for scenario_dir in category_dir.iterdir():
                    if scenario_dir.is_dir():
                        img_dir = scenario_dir / 'images'
                        ann_dir = scenario_dir / 'annotations'

                        if img_dir.exists():
                            for img_path in img_dir.glob('*.png'):
                                ann_path = ann_dir / f"{img_path.stem}.json"

                                self.image_paths.append(str(img_path))
                                self.annotation_paths.append(
                                    str(ann_path) if ann_path.exists() else None
                                )

                                # Load metadata if available
                                meta_path = scenario_dir / 'metadata.json'
                                if meta_path.exists():
                                    with open(meta_path, 'r') as f:
                                        meta = json.load(f)
                                else:
                                    meta = {}

                                self.metadata.append({
                                    'dataset': 'edge_cases',
                                    'category': category_dir.name,
                                    'scenario': scenario_dir.name,
                                    'weather': meta.get('weather', 'unknown'),
                                    'time_of_day': meta.get('time_of_day', 'unknown'),
                                    'scene_type': 'edge_case',
                                    'difficulty': meta.get('difficulty', 0.8)
                                })
    
    # --- MISSING METHODS ADDED BELOW ---

    def _create_transforms(self) -> transforms.Compose:
        """Creates the image transformation pipeline."""
        transform_list = [transforms.Resize(self.image_size)]

        if self.augment:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            transform_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))

        transform_list.append(transforms.ToTensor())

        if self.normalize:
            # Normalize to [-1, 1] for GANs
            transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any, Dict]:
        """Fetches a single data sample."""
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        transformed_image = self.transform(image)

        # Note: Annotations are not transformed in this example.
        # This would require more complex logic depending on the annotation type
        # (e.g., resizing segmentation masks with nearest-neighbor interpolation).
        annotation = self.annotation_paths[idx]
        meta = self.metadata[idx]

        return transformed_image, annotation, meta

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DrivingSceneDataset

        Args:
            batch: List of tuples (image, annotation, metadata)

        Returns:
            dict with:
                - 'images': Tensor of stacked images
                - 'annotations': List (some entries may be None)
                - 'metadata': List of metadata dicts
        """
        images, annotations, metadata = zip(*batch)

        return {
            'images': torch.stack(images),
            'annotations': list(annotations),  # can include None
            'metadata': list(metadata)
        }
