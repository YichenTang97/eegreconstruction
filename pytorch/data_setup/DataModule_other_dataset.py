import numpy as np
import os
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from typing import Literal
from sklearn.preprocessing import LabelEncoder

from pytorch.data_setup.Dataset_other_dataset import Dataset_Other

class DataModule_Other(pl.LightningDataModule):
    """
    DataModule for preprocessed data saved as individual npy files.
    
    This expects data in separate files:
    - X_train.npy, X_val.npy, X_test.npy: shape (n_images, n_repeats, n_channels, n_timestamps)
    - y_train.npy, y_val.npy, y_test.npy: shape (n_images,) - main labels (can be strings or integers)
    - y_cls_train.npy, y_cls_val.npy, y_cls_test.npy: shape (n_images,) - class labels (can be strings or integers)
    - ch_names.npy: channel names

    Args:
        data_dir: Path to directory containing the individual npy files
        label: Whether to use "label" (main labels) or "class" (class labels)
        val: Whether to use separate validation data (True) or concatenate train+val (False)
        batch_size: Batch size for training and validation
        num_workers: Number of workers for the dataloader
        **kwargs: Additional arguments (for compatibility)
    """
    def __init__(
            self, 
            data_dir: str, 
            label: Literal["label", "class"] = "label",
            val: bool = True,
            batch_size: int = 32, 
            num_workers: int = 0, 
            **kwargs):
        
        super().__init__()
        self.data_dir = data_dir
        self.label = label
        self.val = val
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create a shared label encoder by examining all splits
        self.label_encoder = self._create_shared_label_encoder()
        
        # Initialize datasets to get metadata
        temp_dataset = Dataset_Other(
            Path(self.data_dir), 
            label=self.label, 
            split="train",
            label_encoder=self.label_encoder
        )
        self.dims = (temp_dataset.n_channels, temp_dataset.n_times)
        self.num_classes = temp_dataset.n_classes
        self.ch_names = temp_dataset.ch_names
        self.label_names = temp_dataset.get_label_names()
        del temp_dataset  # Clean up temporary dataset
    
    def _create_shared_label_encoder(self):
        """Create a label encoder that's consistent across all splits"""
        data_dir = Path(self.data_dir)
        
        # Determine label file prefix
        if self.label == "label":
            label_prefix = "y"
        else:
            label_prefix = "y_cls"
        
        # Collect all labels from all splits to ensure consistent encoding
        all_labels = []
        for split in ["train", "val", "test"]:
            label_file = data_dir / f'{label_prefix}_{split}.npy'
            if label_file.exists():
                labels = np.load(label_file, allow_pickle=True)
                all_labels.extend(labels.flatten())
        
        # Check if labels are strings
        all_labels = np.array(all_labels)
        if all_labels.dtype.kind in ['U', 'S', 'O']:  # String labels
            encoder = LabelEncoder()
            encoder.fit(all_labels)
            return encoder
        else:
            # Numeric labels, no encoder needed
            return None
       
    def setup(self, stage=None):
        """Loads data and prepares PyTorch tensor datasets for each split."""
        if stage == "fit" or stage is None:
            if self.val:
                # Use separate train and validation datasets
                self.train_dataset = Dataset_Other(
                    Path(self.data_dir), 
                    label=self.label, 
                    split="train",
                    label_encoder=self.label_encoder
                )
                self.val_dataset = Dataset_Other(
                    Path(self.data_dir), 
                    label=self.label, 
                    split="val",
                    label_encoder=self.label_encoder
                )
            else:
                # Concatenate train and validation datasets for training
                train_dataset = Dataset_Other(
                    Path(self.data_dir), 
                    label=self.label, 
                    split="train",
                    label_encoder=self.label_encoder
                )
                val_dataset = Dataset_Other(
                    Path(self.data_dir), 
                    label=self.label, 
                    split="val",
                    label_encoder=self.label_encoder
                )
                # Combine both datasets for training
                self.train_dataset = ConcatDataset([train_dataset, val_dataset])
                # Use the been combined validation dataset for validation (though validation won't be meaningful)
                self.val_dataset = val_dataset
            
        if stage == "test" or stage is None:
            self.test_dataset = Dataset_Other(
                Path(self.data_dir), 
                label=self.label, 
                split="test",
                label_encoder=self.label_encoder
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self):
        if self.val:
            # Return separate validation dataloader
            return DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=False
            )
        else:
            # When val=False, validation doesn't make sense, but PyTorch Lightning requires it
            # Return a small subset of training data for validation (just to satisfy PL)
            return DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=False
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )

    def predict_dataloader(self):
        # For prediction, use test dataset without shuffling
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )
    
    def get_dataset_info(self):
        """Get information about all dataset splits."""
        info = {}
        for split in ["train", "val", "test"]:
            dataset = Dataset_Other(
                Path(self.data_dir), 
                label=self.label, 
                split=split,
                label_encoder=self.label_encoder
            )
            info[split] = dataset.get_info()
        
        # Add information about the current configuration
        info['config'] = {
            'val_separate': self.val,
            'effective_train_samples': info['train']['n_samples'] + (info['val']['n_samples'] if not self.val else 0),
            'effective_val_samples': info['val']['n_samples'] if self.val else info['train']['n_samples'] + info['val']['n_samples']
        }
        return info
    
    def decode_labels(self, encoded_labels):
        """Decode integer labels back to original strings"""
        if hasattr(self, 'train_dataset'):
            # Handle both individual datasets and ConcatDataset
            if hasattr(self.train_dataset, 'decode_labels'):
                return self.train_dataset.decode_labels(encoded_labels)
            elif hasattr(self.train_dataset, 'datasets') and len(self.train_dataset.datasets) > 0:
                # ConcatDataset case - use the first dataset's decoder
                return self.train_dataset.datasets[0].decode_labels(encoded_labels)
        elif self.label_encoder is not None:
            return self.label_encoder.inverse_transform(encoded_labels)
        else:
            return encoded_labels
    
    def get_label_names(self):
        """Get the original string label names"""
        return self.label_names 