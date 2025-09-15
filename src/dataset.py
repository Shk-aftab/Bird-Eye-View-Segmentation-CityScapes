"""
BEV Dataset for Semantic Segmentation
Simplified class mapping focusing on drivable areas and obstacles
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from .temporal_fusion import TemporalDataset


class BEVDataset(Dataset):
    """
    BEV dataset for semantic segmentation
        Simplified classes for autonomous driving
    """
    
    def __init__(self, data_root, split='train', image_size=(384, 384), bev_size=(256, 256), data_percentage=1.0):
        """
        Args:
            data_root: Path to dataset root
            split: 'train' or 'val'
            image_size: (height, width) for camera images
            bev_size: (height, width) for BEV labels
            data_percentage: Percentage of data to use (1.0 = 100%, 0.5 = 50%, etc.)
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.bev_size = bev_size
        self.data_percentage = data_percentage
        
        # Camera views
        self.camera_views = ['front', 'left', 'rear', 'right']
        
        # Get all sample IDs from BEV directory
        bev_dir = os.path.join(data_root, split, 'bev', 'bev')
        all_sample_ids = self._get_sample_ids(bev_dir)
        
        # Apply data percentage
        if split == 'train' and data_percentage < 1.0:
            num_samples = int(len(all_sample_ids) * data_percentage)
            self.sample_ids = all_sample_ids[:num_samples]
            print(f"Using {data_percentage*100:.1f}% of training data: {len(self.sample_ids)}/{len(all_sample_ids)} samples")
        else:
            self.sample_ids = all_sample_ids
            print(f"Found {len(self.sample_ids)} samples in {split} split")
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # BEV label transform (no normalization for segmentation labels)
        self.bev_transform = transforms.Compose([
            transforms.Resize(bev_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # SIMPLIFIED color to class mapping for autonomous driving
        # Merge pole and traffic sign into unlabeled to reduce imbalance
        self.color_to_class = {
            (0, 0, 0): 0,           # unlabeled (includes pole, traffic sign)
            (0, 0, 142): 1,         # car (obstacle)
            (107, 142, 35): 2,      # vegetation (obstacle)
            (128, 64, 128): 3,      # road (drivable)
            (152, 251, 152): 4,     # terrain (drivable)
            (153, 153, 153): 0,     # pole -> unlabeled
            (180, 165, 180): 5,     # guard rail (obstacle)
            (220, 220, 0): 0,       # traffic sign -> unlabeled
            (244, 35, 232): 6,      # sidewalk (obstacle)
        }
        
        # Number of classes for autonomous driving
        self.num_classes = 7  # 0:unlabeled, 1:car, 2:vegetation, 3:road, 4:terrain, 5:guard_rail, 6:sidewalk
        self.class_names = {
            0: "unlabeled",
            1: "car", 
            2: "vegetation",
            3: "road",
            4: "terrain", 
            5: "guard_rail",
            6: "sidewalk"
        }
        
        print(f"Dataset for autonomous driving:")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Class mapping: {self.class_names}")
        print(f"  - Pole and traffic sign merged into unlabeled")
        
    def _get_sample_ids(self, bev_dir):
        """Extract sample IDs from BEV directory"""
        if not os.path.exists(bev_dir):
            raise ValueError(f"BEV directory not found: {bev_dir}")
            
        pattern = os.path.join(bev_dir, "*.png")
        files = glob.glob(pattern)
        
        sample_ids = []
        for file_path in files:
            filename = os.path.basename(file_path)
            sample_id = os.path.splitext(filename)[0]
            sample_ids.append(sample_id)
            
        return sorted(sample_ids)
    
    def _load_camera_images(self, sample_id):
        """Load all camera view images for a sample"""
        images = {}
        
        for view in self.camera_views:
            img_path = os.path.join(
                self.data_root, self.split, view, view, f"{sample_id}.png"
            )
            
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                images[view] = self.image_transform(image)
            else:
                # Create zero tensor if image doesn't exist
                images[view] = torch.zeros(3, *self.image_size)
                print(f"Warning: Missing {view} image for sample {sample_id}")
                
        return images
    
    def _rgb_to_class_tensor(self, rgb_image):
        """
        Convert RGB image to simplified class ID tensor for autonomous driving
        Args:
            rgb_image: PIL Image or numpy array (H, W, 3)
        Returns:
            class_tensor: torch.Tensor (H, W) with class IDs
        """
        if isinstance(rgb_image, Image.Image):
            rgb_array = np.array(rgb_image)
        else:
            rgb_array = rgb_image
        
        h, w, c = rgb_array.shape
        class_tensor = torch.zeros(h, w, dtype=torch.long)
        
        for i in range(h):
            for j in range(w):
                pixel_color = tuple(rgb_array[i, j])
                if pixel_color in self.color_to_class:
                    class_tensor[i, j] = self.color_to_class[pixel_color]
                else:
                    class_tensor[i, j] = 0  # Default to unlabeled
        
        return class_tensor
    
    def _load_bev_label(self, sample_id):
        """Load BEV ground truth label with simplified class mapping"""
        bev_path = os.path.join(
            self.data_root, self.split, 'bev', 'bev', f"{sample_id}.png"
        )
        
        if not os.path.exists(bev_path):
            raise ValueError(f"BEV label not found: {bev_path}")
            
        # Load as RGB for proper color mapping
        bev_image = Image.open(bev_path).convert('RGB')
        
        # Resize if needed
        if bev_image.size != self.bev_size:
            bev_image = bev_image.resize(self.bev_size, Image.NEAREST)
        
        # Convert RGB to simplified class IDs
        class_tensor = self._rgb_to_class_tensor(bev_image)
        
        return class_tensor
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load camera images
        camera_images = self._load_camera_images(sample_id)
        
        # Load BEV label with simplified class mapping
        bev_label = self._load_bev_label(sample_id)
        
        return {
            'sample_id': sample_id,
            'camera_images': camera_images,
            'bev_label': bev_label
        }


def create_dataloader(data_root, split='train', batch_size=4, num_workers=2, data_percentage=1.0, 
                     use_temporal=False, temporal_config=None, img_height=384, img_width=384, 
                     bev_height=256, bev_width=256, **kwargs):
    """Create DataLoader for BEV dataset"""
    # Create base dataset with config dimensions
    base_dataset = BEVDataset(
        data_root, 
        split=split, 
        data_percentage=data_percentage,
        image_size=(img_height, img_width),
        bev_size=(bev_height, bev_width),
        **kwargs
    )
    
    # Wrap with temporal dataset if enabled
    if use_temporal and temporal_config:
        num_timesteps = temporal_config.get('num_timesteps', 3)
        timestep_interval = temporal_config.get('timestep_interval', 500)
        
        dataset = TemporalDataset(
            base_dataset=base_dataset,
            num_timesteps=num_timesteps,
            timestep_interval=timestep_interval
        )
        print(f"  - Temporal dataset enabled: {num_timesteps} timesteps, {timestep_interval}ms interval")
    else:
        dataset = base_dataset
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


def test_dataset():
    """Test the simplified dataset loader"""
    print("Testing BEV Dataset for Semantic Segmentation...")
    
    data_root = "data/cam2bev-data-master-1_FRLR/1_FRLR"
    
    # Test train dataset
    print("\n1. Testing train dataset...")
    train_loader = create_dataloader(data_root, split='train', batch_size=2)
    
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Sample IDs: {batch['sample_id']}")
        print(f"  Camera images shape: {[v.shape for v in batch['camera_images'].values()]}")
        print(f"  BEV label shape: {batch['bev_label'].shape}")
        print(f"  BEV label unique values: {torch.unique(batch['bev_label'])}")
        print(f"  BEV label class distribution:")
        
        # Show class distribution for this batch
        unique_classes, counts = torch.unique(batch['bev_label'], return_counts=True)
        total_pixels = batch['bev_label'].numel()
        class_names = {0: "unlabeled", 1: "car", 2: "vegetation", 3: "road", 4: "terrain", 5: "guard_rail", 6: "sidewalk"}
        for class_id, count in zip(unique_classes, counts):
            percentage = (count.item() / total_pixels) * 100
            class_name = class_names.get(class_id.item(), f"class_{class_id.item()}")
            print(f"    Class {class_id.item()} ({class_name:12s}): {count.item():6d} pixels ({percentage:5.2f}%)")
        
        if i >= 2:  # Test only first 3 batches
            break
    
    print("\n2. Testing validation dataset...")
    val_loader = create_dataloader(data_root, split='val', batch_size=2)
    
    for i, batch in enumerate(val_loader):
        print(f"Val Batch {i}:")
        print(f"  Sample IDs: {batch['sample_id']}")
        print(f"  BEV label unique values: {torch.unique(batch['bev_label'])}")
        
        if i >= 1:  # Test only first 2 batches
            break
    
    print("\nDataset test completed!")


if __name__ == "__main__":
    test_dataset()
