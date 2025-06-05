#!/usr/bin/env python3
"""
Test script for the new dataset classes
"""
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

try:
    from pytorch.data_setup.Dataset_other_dataset import Dataset_Other
    from pytorch.data_setup.DataModule_other_dataset import DataModule_Other
    
    print("✓ Successfully imported Dataset_Other and DataModule_Other")
    
    # Test dataset loading
    data_dir = "./experiments/experiment_gtec_coco_compact_250527/data"
    
    print(f"\nTesting dataset loading from: {data_dir}")
    
    # Test DataModule
    dm = DataModule_Other(
        data_dir=data_dir,
        label="label",
        batch_size=32,
        num_workers=0
    )
    
    print(f"✓ DataModule created successfully")
    print(f"  - Data dimensions: {dm.dims}")
    print(f"  - Number of classes: {dm.num_classes}")
    print(f"  - Channel names: {dm.ch_names}")
    
    # Get dataset info
    dataset_info = dm.get_dataset_info()
    print(f"\nDataset Information:")
    for split, info in dataset_info.items():
        print(f"  {split.upper()}:")
        print(f"    - Samples: {info['n_samples']}")
        print(f"    - Shape: {info['data_shape']}")
        print(f"    - Classes: {info['n_classes']}")
    
    print("\n✓ All tests passed! The dataset classes are working correctly.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except FileNotFoundError as e:
    print(f"✗ File not found: {e}")
    print("Make sure the data file exists at the specified path")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()