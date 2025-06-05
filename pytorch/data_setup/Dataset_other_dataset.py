import numpy as np
from pathlib import Path
from typing import Literal
import torch
from sklearn.preprocessing import LabelEncoder

class Dataset_Other():
    """
    Pytorch Dataset for preprocessed data saved as individual npy files.
    
    This expects data in separate files:
    - X_train.npy, X_val.npy, X_test.npy: shape (n_images, n_repeats, n_channels, n_timestamps)
    - y_train.npy, y_val.npy, y_test.npy: shape (n_images,) - main labels (can be strings or integers)
    - y_cls_train.npy, y_cls_val.npy, y_cls_test.npy: shape (n_images,) - class labels (can be strings or integers)
    - ch_names.npy: channel names
    
    For training: flattens first two dimensions -> (n_images x n_repeats, n_channels, n_timestamps)
    For validation/test: averages across repeats -> (n_images, n_channels, n_timestamps)

    Args:
        data_dir: Path to directory containing the individual npy files
        label: Whether to use "label" (main labels) or "class" (class labels)
        split: Whether to use "train", "val", or "test" split
        label_encoder: Optional pre-fitted LabelEncoder for consistent encoding across splits
    """
    def __init__(
            self, 
            data_dir: Path, 
            label: Literal["label", "class"] = "label",
            split: Literal["train", "val", "test"] = "train",
            label_encoder: LabelEncoder = None
            ):
        if label not in ["label", "class"]:
            raise ValueError("label must be either 'label' or 'class'")
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be either 'train', 'val', or 'test'")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.split = split
        self.label_type = label
        
        data_dir = Path(data_dir)
        
        # Load data and labels for the specified split
        data_file = data_dir / f'X_{split}.npy'
        if self.label_type == "label":
            label_file = data_dir / f'y_{split}.npy'
        else:
            label_file = data_dir / f'y_cls_{split}.npy'
        
        # Check if files exist
        if not data_file.exists():
            raise FileNotFoundError(f"Could not find {data_file}")
        if not label_file.exists():
            raise FileNotFoundError(f"Could not find {label_file}")
        
        # Load data and labels
        raw_data = np.load(data_file)  # shape: (n_images, n_repeats, n_channels, n_timestamps)
        raw_labels = np.load(label_file, allow_pickle=True)  # shape: (n_images,) - can be strings
        
        # Handle string labels by encoding them to integers
        if raw_labels.dtype.kind in ['U', 'S', 'O']:  # Unicode, bytes, or object (string) arrays
            if label_encoder is None:
                # Create new label encoder
                self.label_encoder = LabelEncoder()
                # Fit on all unique labels in this split
                self.label_encoder.fit(raw_labels)
                self.is_encoder_owner = True
            else:
                # Use provided label encoder
                self.label_encoder = label_encoder
                self.is_encoder_owner = False
            
            # Store original string labels for reference
            self.original_labels = raw_labels.copy()
            # Encode to integers
            encoded_labels = self.label_encoder.transform(raw_labels)
            self.label_names = self.label_encoder.classes_
        else:
            # Labels are already numeric
            encoded_labels = raw_labels
            self.original_labels = raw_labels.copy()
            self.label_encoder = None
            self.is_encoder_owner = False
            self.label_names = np.unique(raw_labels)
        
        if split == "train":
            # For training: flatten first two dimensions
            n_images, n_repeats, n_channels, n_timestamps = raw_data.shape
            self.data = raw_data.reshape(n_images * n_repeats, n_channels, n_timestamps)
            # Repeat labels for each repeat
            self.labels = np.repeat(encoded_labels, n_repeats)
            if hasattr(self, 'original_labels'):
                self.original_labels_repeated = np.repeat(self.original_labels, n_repeats)
        else:
            # For validation/test: average across repeats
            self.data = np.mean(raw_data, axis=1)  # shape: (n_images, n_channels, n_timestamps)
            self.labels = encoded_labels
            if hasattr(self, 'original_labels'):
                self.original_labels_repeated = self.original_labels
        
        # Convert to torch tensors
        self.data = torch.from_numpy(self.data).float().to(self.device)
        self.labels = torch.from_numpy(self.labels).long().to(self.device)
        
        # Load channel names if available
        ch_names_file = data_dir / 'ch_names.npy'
        if ch_names_file.exists():
            self.ch_names = np.load(ch_names_file, allow_pickle=True).tolist()
        else:
            self.ch_names = None
            
        # Store metadata
        self.n_channels = self.data.shape[1]
        self.n_times = self.data.shape[2]
        self.n_classes = len(np.unique(self.labels.cpu().numpy()))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_label_names(self):
        """Get the original string label names"""
        return self.label_names
    
    def decode_labels(self, encoded_labels):
        """Decode integer labels back to original strings"""
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(encoded_labels)
        else:
            return encoded_labels
    
    def get_info(self):
        """Return dataset information"""
        return {
            'n_samples': len(self),
            'n_channels': self.n_channels,
            'n_times': self.n_times,
            'n_classes': self.n_classes,
            'ch_names': self.ch_names,
            'data_shape': self.data.shape,
            'split': self.split,
            'label_type': self.label_type,
            'label_names': self.label_names.tolist() if hasattr(self.label_names, 'tolist') else self.label_names,
            'has_string_labels': self.label_encoder is not None
        } 