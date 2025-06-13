#!/usr/bin/env python3
"""
SafeScene AI - Dataset Download Script
Downloads and prepares Cityscapes and KITTI datasets for training.
File: scripts/download_datasets.py
"""

import os
import sys
import argparse
import zipfile
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
import json
import shutil

class DatasetDownloader:
    def __init__(self, data_root="./data"):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        
        # Dataset configurations
        self.datasets = {
            "cityscapes": {
                "description": "Cityscapes urban driving dataset",
                "sample_url": "https://www.cityscapes-dataset.com/file-handling/?packageID=3",
                "full_info": "Please register at https://www.cityscapes-dataset.com to download full dataset",
                "structure": ["leftImg8bit", "gtFine"],
                "size": "11GB"
            },
            "kitti": {
                "description": "KITTI autonomous driving dataset",
                "sample_url": "https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d",
                "full_info": "Visit https://www.cvlibs.net/datasets/kitti/ for full dataset",
                "structure": ["image_2", "label_2", "calib"],
                "size": "12GB"
            }
        }
    
    def download_file(self, url, destination, description="Downloading"):
        """Download file with progress bar"""
        try:
            response = urllib.request.urlopen(url)
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(destination, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def create_sample_dataset(self, dataset_name):
        """Create sample dataset structure for development"""
        print(f"\n📦 Creating sample {dataset_name} dataset structure...")
        
        dataset_path = self.raw_dir / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        if dataset_name == "cityscapes":
            self._create_cityscapes_sample(dataset_path)
        elif dataset_name == "kitti":
            self._create_kitti_sample(dataset_path)
        
        # Create dataset info file
        info = {
            "dataset": dataset_name,
            "type": "sample",
            "description": f"Sample {dataset_name} dataset for development",
            "note": "Replace with full dataset for production training"
        }
        
        info_file = dataset_path / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"✅ Sample {dataset_name} structure created!")
    
    def _create_cityscapes_sample(self, dataset_path):
        """Create Cityscapes sample structure"""
        # Create directory structure
        splits = ["train", "val", "test"]
        cities = ["aachen", "berlin", "cologne"]
        
        for split in splits:
            # Images
            img_dir = dataset_path / "leftImg8bit" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            
            # Annotations
            ann_dir = dataset_path / "gtFine" / split
            ann_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sample structure
            for city in cities[:2 if split == "train" else 1]:
                city_img_dir = img_dir / city
                city_ann_dir = ann_dir / city
                city_img_dir.mkdir(exist_ok=True)
                city_ann_dir.mkdir(exist_ok=True)
                
                # Create placeholder files
                for i in range(3):
                    img_name = f"{city}_000000_00000{i}_leftImg8bit.png"
                    ann_name = f"{city}_000000_00000{i}_gtFine_labelIds.png"
                    
                    (city_img_dir / img_name).touch()
                    (city_ann_dir / ann_name).touch()
    
    def _create_kitti_sample(self, dataset_path):
        """Create KITTI sample structure"""
        # Create directory structure
        subdirs = ["image_2", "label_2", "calib", "velodyne"]
        
        for subdir in subdirs:
            dir_path = dataset_path / "training" / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create sample files
            for i in range(10):
                if subdir == "image_2":
                    filename = f"{i:06d}.png"
                elif subdir == "label_2":
                    filename = f"{i:06d}.txt"
                elif subdir == "calib":
                    filename = f"{i:06d}.txt"
                elif subdir == "velodyne":
                    filename = f"{i:06d}.bin"
                
                (dir_path / filename).touch()
    
    def prepare_processed_structure(self):
        """Create processed data directory structure"""
        print("\n📁 Creating processed data structure...")
        
        splits = ["train", "val", "test"]
        subdirs = ["images", "annotations", "metadata"]
        
        for split in splits:
            for subdir in subdirs:
                dir_path = self.processed_dir / split / subdir
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create processing config
        config = {
            "image_size": [1024, 512],
            "normalization": "imagenet",
            "augmentations": ["horizontal_flip", "color_jitter", "random_crop"],
            "format": "png"
        }
        
        config_file = self.processed_dir / "processing_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        print("✅ Processed data structure created!")
    
    def create_custom_edge_cases_structure(self):
        """Create structure for custom edge cases"""
        print("\n🎯 Creating custom edge cases structure...")
        
        edge_cases_dir = self.raw_dir / "custom_edge_cases"
        edge_cases_dir.mkdir(parents=True, exist_ok=True)
        
        # Categories of edge cases
        categories = {
            "weather_extreme": ["heavy_rain", "dense_fog", "snow_storm"],
            "lighting_challenge": ["sun_glare", "night_rain", "tunnel_exit"],
            "object_anomaly": ["jaywalking", "wrong_way_driver", "debris"],
            "infrastructure": ["construction_zone", "emergency_vehicle", "accident_scene"]
        }
        
        for category, scenarios in categories.items():
            cat_dir = edge_cases_dir / category
            cat_dir.mkdir(exist_ok=True)
            
            for scenario in scenarios:
                scenario_dir = cat_dir / scenario
                scenario_dir.mkdir(exist_ok=True)
                
                # Create placeholder for images and annotations
                (scenario_dir / "images").mkdir(exist_ok=True)
                (scenario_dir / "annotations").mkdir(exist_ok=True)
                (scenario_dir / "metadata.json").touch()
        
        # Create edge cases catalog
        catalog = {
            "total_categories": len(categories),
            "categories": categories,
            "description": "Custom edge cases for autonomous vehicle safety testing",
            "collection_guidelines": {
                "image_format": "PNG or JPEG",
                "min_resolution": "1920x1080",
                "annotation_format": "COCO or YOLO",
                "metadata_required": ["weather", "lighting", "difficulty_score"]
            }
        }
        
        catalog_file = edge_cases_dir / "edge_cases_catalog.json"
        with open(catalog_file, 'w') as f:
            json.dump(catalog, f, indent=4)
        
        print("✅ Custom edge cases structure created!")
    
    def download_dataset(self, dataset_name):
        """Download or prepare dataset"""
        if dataset_name not in self.datasets:
            print(f"❌ Unknown dataset: {dataset_name}")
            return False
        
        dataset_info = self.datasets[dataset_name]
        print(f"\n🌐 Dataset: {dataset_info['description']}")
        print(f"📊 Full dataset size: {dataset_info['size']}")
        print(f"🔗 {dataset_info['full_info']}")
        
        # For this demo, create sample structure
        self.create_sample_dataset(dataset_name)
        
        print(f"\n💡 Sample {dataset_name} dataset created for development.")
        print("   For production training, please download the full dataset from official sources.")
        
        return True
    
    def create_dataset_splits(self):
        """Create train/val/test split configuration"""
        print("\n📊 Creating dataset split configuration...")
        
        split_config = {
            "train": 0.7,
            "val": 0.15,
            "test": 0.15,
            "random_seed": 42,
            "stratify": True,
            "notes": "Splits will be created during data processing phase"
        }
        
        config_file = self.data_root / "split_config.json"
        with open(config_file, 'w') as f:
            json.dump(split_config, f, indent=4)
        
        print("✅ Split configuration created!")
    
    def verify_datasets(self):
        """Verify dataset structure"""
        print("\n🔍 Verifying dataset structure...")
        
        datasets_found = []
        for dataset_name in self.datasets.keys():
            dataset_path = self.raw_dir / dataset_name
            if dataset_path.exists():
                datasets_found.append(dataset_name)
                print(f"✅ {dataset_name}: Found")
            else:
                print(f"❌ {dataset_name}: Not found")
        
        if self.raw_dir / "custom_edge_cases" in self.raw_dir.iterdir():
            print("✅ custom_edge_cases: Found")
        
        return datasets_found
    
    def run(self, datasets_to_download):
        """Run dataset download process"""
        print("🚀 SafeScene AI - Dataset Download")
        print("=" * 50)
        
        # Create base structure
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Download requested datasets
        for dataset in datasets_to_download:
            self.download_dataset(dataset)
        
        # Create additional structures
        self.prepare_processed_structure()
        self.create_custom_edge_cases_structure()
        self.create_dataset_splits()
        
        # Verify everything
        self.verify_datasets()
        
        print("\n✨ Dataset preparation complete!")
        print("\n📋 Next steps:")
        print("1. For full datasets, download from official sources")
        print("2. Place downloaded data in respective raw/ directories")
        print("3. Run data processing pipeline to create training data")

def main():
    parser = argparse.ArgumentParser(description="SafeScene AI Dataset Downloader")
    parser.add_argument("--dataset", action="append", 
                       choices=["cityscapes", "kitti", "all"],
                       help="Datasets to download")
    parser.add_argument("--data-root", type=str, default="./data",
                       help="Root directory for datasets")
    
    args = parser.parse_args()
    
    # Default to all datasets if none specified
    if not args.dataset:
        datasets = ["cityscapes", "kitti"]
    elif "all" in args.dataset:
        datasets = ["cityscapes", "kitti"]
    else:
        datasets = list(set(args.dataset))
    
    downloader = DatasetDownloader(args.data_root)
    downloader.run(datasets)

if __name__ == "__main__":
    main()