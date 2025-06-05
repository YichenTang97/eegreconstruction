import numpy as np
import os
from einops import rearrange
import torch
from pathlib import Path
import torchvision.transforms as transforms
from typing import Callable, Literal
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def identity(x):
    return x

def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x
    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(img):
    """Normalize image to [-1, 1] range as expected by diffusion models"""
    # Handle PIL Image input
    if hasattr(img, 'mode'):  # PIL Image
        img = np.array(img) / 255.0
    
    # Ensure channel-first format for processing
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    
    # Convert to tensor and normalize to [-1, 1]
    img = torch.tensor(img, dtype=torch.float32)
    img = img * 2.0 - 1.0
    return img

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
    if img.shape[-1] == 3:
        return rearrange(img, 'h w c -> c h w')
    return img

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

class CustomLabelEncoder:
    """Custom label encoder to handle string labels consistently"""
    def __init__(self):
        self.classes_ = None
        self.class_to_index = None
        
    def fit(self, labels):
        self.classes_ = np.unique(labels)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self
    
    def transform(self, labels):
        if self.class_to_index is None:
            raise ValueError("LabelEncoder has not been fitted yet.")
        return np.array([self.class_to_index[label] for label in labels])
    
    def inverse_transform(self, encoded_labels):
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted yet.")
        return np.array([self.classes_[idx] for idx in encoded_labels])

class EEGDataset_Other():
    """
    EEG Dataset for preprocessed data saved as individual npy files.
    
    This expects data structure:
    experiment_folder/
    ├── data/
    │   ├── X_train.npy, X_val.npy, X_test.npy: shape (n_images, n_repeats, n_channels, n_timestamps)
    │   ├── y_train.npy, y_val.npy, y_test.npy: shape (n_images,) - main labels
    │   ├── y_cls_train.npy, y_cls_val.npy, y_cls_test.npy: shape (n_images,) - class labels
    │   └── ch_names.npy: channel names
    └── image_pool/
        └── [label_name]_[index].jpg  # Images with names matching labels

    For reconstruction tasks:
    - split="train": Combines train and validation data for maximum training data
    - split="test": Uses only test data for final evaluation

    Args:
        experiment_dir: Path to experiment directory containing data/ and image_pool/
        image_transform: Optional transform to be applied on images
        train: Deprecated, use split parameter instead
        split: Which split to use ('train' combines train+val, 'test' for test only)
        label: Whether to use "label" (main labels) or "class" (class labels)
        preload_images: Whether to pre-load all images into memory
        label_encoder: Optional pre-fitted label encoder for consistent encoding
    """
    def __init__(
            self, 
            experiment_dir: Path,
            image_transform: Callable = None,
            train: bool = True, 
            split: Literal["train", "val", "test"] = "train",
            label: Literal["label", "class"] = "class",
            preload_images: bool = False,
            label_encoder: CustomLabelEncoder = None,
            ):
        
        self.experiment_dir = Path(experiment_dir)
        self.data_dir = self.experiment_dir / "data"
        self.image_pool_dir = self.experiment_dir / "image_pool"
        self.preload_images = preload_images
        self.image_transform = image_transform if image_transform else identity
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.split = split
        self.label_type = label
        
        # Load EEG data and labels
        self._load_eeg_data()
        
        # Create image path mapping FIRST (needed for label encoder validation)
        self._create_image_mapping()
        
        # Create or use provided label encoder
        self._setup_label_encoder(label_encoder)
        
        # Optionally preload images
        if self.preload_images:
            self._preload_images()

    def _load_eeg_data(self):
        """Load EEG data and labels from individual npy files"""
        
        if self.split == "train":
            # For reconstruction: combine train and validation data for maximum training data
            print("Loading training data: combining train and validation splits...")
            
            # Load training data
            train_data_file = self.data_dir / 'X_train.npy'
            if not train_data_file.exists():
                raise FileNotFoundError(f"Could not find {train_data_file}")
            train_data = np.load(train_data_file)  # shape: (n_images, n_repeats, n_channels, n_timestamps)
            
            # Load training labels
            if self.label_type == "label":
                train_label_file = self.data_dir / 'y_train.npy'
            else:
                train_label_file = self.data_dir / 'y_cls_train.npy'
            if not train_label_file.exists():
                raise FileNotFoundError(f"Could not find {train_label_file}")
            train_labels = np.load(train_label_file, allow_pickle=True)
            
            # Load validation data
            val_data_file = self.data_dir / 'X_val.npy'
            if not val_data_file.exists():
                raise FileNotFoundError(f"Could not find {val_data_file}")
            val_data = np.load(val_data_file)
            
            # Load validation labels
            if self.label_type == "label":
                val_label_file = self.data_dir / 'y_val.npy'
            else:
                val_label_file = self.data_dir / 'y_cls_val.npy'
            if not val_label_file.exists():
                raise FileNotFoundError(f"Could not find {val_label_file}")
            val_labels = np.load(val_label_file, allow_pickle=True)
            
            # Concatenate train and validation data
            raw_data = np.concatenate([train_data, val_data], axis=0)
            raw_labels = np.concatenate([train_labels, val_labels], axis=0)
            
            print(f"Combined training data: {train_data.shape[0]} train + {val_data.shape[0]} val = {raw_data.shape[0]} total images")
            
            # Flatten first two dimensions for training (more training samples)
            n_images, n_repeats, n_channels, n_timestamps = raw_data.shape
            self.eeg_data = raw_data.reshape(n_images * n_repeats, n_channels, n_timestamps)
            # Repeat labels for each repeat
            self.raw_labels = np.repeat(raw_labels, n_repeats)
            
        else:
            # For test: use only test data
            print(f"Loading {self.split} data...")
            
            # Load EEG data
            data_file = self.data_dir / f'X_{self.split}.npy'
            if not data_file.exists():
                raise FileNotFoundError(f"Could not find {data_file}")
            
            raw_data = np.load(data_file)  # shape: (n_images, n_repeats, n_channels, n_timestamps)
            
            # Load labels
            if self.label_type == "label":
                label_file = self.data_dir / f'y_{self.split}.npy'
            else:
                label_file = self.data_dir / f'y_cls_{self.split}.npy'
            
            if not label_file.exists():
                raise FileNotFoundError(f"Could not find {label_file}")
            
            raw_labels = np.load(label_file, allow_pickle=True)  # shape: (n_images,)
            
            # For test: average across repeats for clean evaluation
            self.eeg_data = np.mean(raw_data, axis=1)  # shape: (n_images, n_channels, n_timestamps)
            self.raw_labels = raw_labels

    def _setup_label_encoder(self, label_encoder):
        """Setup label encoder for string labels"""
        if self.raw_labels.dtype.kind in ['U', 'S', 'O']:  # String labels
            if label_encoder is None:
                # Create new label encoder - need to fit on all possible labels across all splits
                self.label_encoder = CustomLabelEncoder()
                
                # Collect all unique labels from all splits to ensure consistency
                all_labels = set(self.raw_labels)
                
                # Also check what labels are available in image_pool for validation
                image_labels = set(self.image_paths.keys())
                
                # For training split, also load other splits to get all possible labels
                if self.split == "train":
                    for split_name in ["val", "test"]:
                        split_file = self.data_dir / f'y_{split_name}.npy'
                        if split_file.exists():
                            try:
                                split_labels = np.load(split_file, allow_pickle=True)
                                all_labels.update(split_labels)
                            except Exception as e:
                                print(f"Warning: Could not load {split_name} labels: {e}")
                
                # Fit encoder on all unique labels found
                unique_labels = sorted(list(all_labels))
                self.label_encoder.fit(unique_labels)
                
                print(f"Label encoder fitted on {len(unique_labels)} unique labels")
                print(f"Sample labels: {unique_labels[:10] if len(unique_labels) >= 10 else unique_labels}")
                
                # Check for missing images
                missing_images = all_labels - image_labels
                if missing_images:
                    missing_sample = sorted(list(missing_images))[:5]
                    print(f"Warning: {len(missing_images)} labels have no corresponding images")
                    print(f"Sample missing: {missing_sample}")
                    
            else:
                self.label_encoder = label_encoder
            
            # Encode labels to integers
            self.labels = self.label_encoder.transform(self.raw_labels)
            self.label_names = self.label_encoder.classes_
        else:
            # Labels are already numeric
            self.labels = self.raw_labels
            self.label_encoder = None
            self.label_names = np.unique(self.raw_labels)
        
        # Convert to torch tensors
        self.eeg_data = torch.from_numpy(self.eeg_data).to(self.device)
        self.labels = torch.from_numpy(self.labels).long().to(self.device)

    def _create_image_mapping(self):
        """Create mapping from labels to image paths"""
        self.image_paths = {}
        
        # Get all image files in image_pool
        if not self.image_pool_dir.exists():
            raise FileNotFoundError(f"Image pool directory not found: {self.image_pool_dir}")
        
        image_files = list(self.image_pool_dir.glob("*.jpg")) + list(self.image_pool_dir.glob("*.png"))
        
        if not image_files:
            raise ValueError(f"No image files found in {self.image_pool_dir}")
        
        # Create mapping from label names to image paths
        for img_path in image_files:
            # Extract label name from filename
            # Expecting format: labelname_index.jpg (e.g., airplane_5.jpg, stopsign_23.jpg)
            filename_stem = img_path.stem  # Remove extension
            
            # For per-image reconstruction, use the full filename stem as the label
            label_name = filename_stem
            
            if label_name not in self.image_paths:
                self.image_paths[label_name] = []
            self.image_paths[label_name].append(img_path)
        
        # Sort image paths for each label for consistency
        for label_name in self.image_paths:
            self.image_paths[label_name].sort()
        
        # Print mapping summary for debugging
        print(f"Created image mapping: {len(self.image_paths)} unique labels found in image_pool")
        if len(self.image_paths) > 0:
            sample_labels = list(self.image_paths.keys())[:5]
            print(f"Sample image labels: {sample_labels}")
        
        # Check for any duplicate mappings (multiple images per label)
        multi_image_labels = {k: len(v) for k, v in self.image_paths.items() if len(v) > 1}
        if multi_image_labels:
            print(f"Labels with multiple images: {multi_image_labels}")
            print("Note: Using first image for each label in reconstruction")

    def _preload_images(self):
        """Preload all images into memory"""
        self.preloaded_images = {}
        
        for label_name, img_paths in self.image_paths.items():
            self.preloaded_images[label_name] = []
            for img_path in img_paths:
                image_raw = Image.open(img_path).convert("RGB")
                image = np.array(image_raw) / 255.0
                transformed_image = self.image_transform(image)
                self.preloaded_images[label_name].append(transformed_image)

    def _get_image_for_label(self, label_idx):
        """Get image for a given label index"""
        if self.label_encoder is not None:
            # Convert back to original label name
            label_name = self.label_encoder.inverse_transform([label_idx])[0]
        else:
            label_name = str(label_idx)
        
        # For per-image reconstruction, we need exact label-to-image matching
        if label_name not in self.image_paths:
            # Check what images are actually available for debugging
            available_labels = list(self.image_paths.keys())
            available_sample = available_labels[:5] if available_labels else []
            
            raise ValueError(
                f"No image found for label '{label_name}'. "
                f"For per-image reconstruction, each label must have a corresponding image. "
                f"Available image labels (showing first 5): {available_sample}. "
                f"Total available: {len(available_labels)}. "
                f"Check that image_pool contains an image with filename pattern matching '{label_name}'."
            )
        
        img_paths = self.image_paths[label_name]
        
        if self.preload_images:
            # Return preloaded image (use first image for this label)
            return self.preloaded_images[label_name][0]
        else:
            # Load image on demand (use first image for this label)
            img_path = img_paths[0]
            image_raw = Image.open(img_path).convert("RGB")
            
            # Apply transforms if provided
            if self.image_transform:
                # The transforms expect PIL Image, then convert to numpy/tensor as needed
                return self.image_transform(image_raw)
            else:
                # Convert to numpy array if no transforms
                image = np.array(image_raw) / 255.0
                return image

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Get EEG data and corresponding image"""
        eeg_data = self.eeg_data[idx].float()
        label_idx = self.labels[idx].item()
        image = self._get_image_for_label(label_idx)
        
        return {'eeg': eeg_data, 'image': image}
    
    def get_label_names(self):
        """Get the original string label names"""
        return self.label_names
    
    def get_info(self):
        """Return dataset information"""
        return {
            'n_samples': len(self),
            'n_channels': self.eeg_data.shape[1],
            'n_times': self.eeg_data.shape[2],
            'n_classes': len(self.label_names),
            'split': self.split,
            'label_type': self.label_type,
            'label_names': self.label_names.tolist() if hasattr(self.label_names, 'tolist') else self.label_names,
            'has_string_labels': self.label_encoder is not None,
            'n_images_per_label': {name: len(paths) for name, paths in self.image_paths.items()}
        } 