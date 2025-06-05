# EEG Visual Reconstruction - Other Dataset

This implementation adapts the EEG visual reconstruction pipeline to work with a new dataset format, specifically designed for per-image EEG visual reconstruction using double conditioning LDM.

## Prerequisites

### Required: Pretrained EEG Classification Model
**🚨 IMPORTANT**: You **must** have a pretrained EEG model that was trained for classification before running this reconstruction pipeline. This is typically:
- An EEG encoder model (e.g., EEGNet, EEG-Transformer) trained to classify images from EEG signals
- A model checkpoint (.ckpt file) and optionally a config file (.yaml)
- The model should be trained on the same or similar EEG data format (channels, sampling rate, etc.)

Without a pretrained EEG classification model, the reconstruction will not work properly.

## Overview

The scripts in this folder are specifically designed to work with experiment datasets that have:
- Individual `.npy` files for EEG data and labels
- An `image_pool` directory containing the corresponding images
- String-based labels that need encoding
- **Per-image reconstruction**: Uses individual image labels (not class labels) for fine-grained reconstruction

## File Structure

### Core Files
- `stageB_ldm_finetune_other.py` - Main training script adapted for the new dataset format
- `dataset_other.py` - Dataset class for loading the new data format
- `config_other.py` - Configuration class for the new dataset
- `test_dataset_other.py` - Test script to verify dataset implementation
- `run_example.py` - Example script showing usage

### Expected Data Structure
```
experiments/experiment_name/
├── data/
│   ├── X_train.npy          # EEG data: (n_images, n_repeats, n_channels, n_timestamps)
│   ├── X_val.npy            # EEG data: (n_images, n_repeats, n_channels, n_timestamps)  
│   ├── X_test.npy           # EEG data: (n_images, n_repeats, n_channels, n_timestamps)
│   ├── y_train.npy          # Per-image labels: (n_images,) - used for reconstruction
│   ├── y_val.npy            # Per-image labels: (n_images,) - used for reconstruction
│   ├── y_test.npy           # Per-image labels: (n_images,) - used for reconstruction
│   ├── y_cls_train.npy      # Class labels: (n_images,) - not used for reconstruction
│   ├── y_cls_val.npy        # Class labels: (n_images,) - not used for reconstruction
│   ├── y_cls_test.npy       # Class labels: (n_images,) - not used for reconstruction
│   └── ch_names.npy         # Channel names
├── image_pool/
│   ├── image_label_0.jpg    # Images with names matching per-image labels
│   ├── image_label_1.jpg
│   ├── another_label_0.jpg
│   └── ...
├── pretrains/               # Pretrained models for this experiment
│   ├── final-model.ckpt     # Pretrained EEG classification model
│   ├── model_config.yaml    # Model configuration
│   └── ...
└── results/                 # All outputs from reconstruction training
    └── reconstruction/
        ├── run_01-01-2024-10-30-15/  # Timestamped training runs
        │   ├── samples_train.png      # Generated training samples
        │   ├── samples_test.png       # Generated test samples
        │   ├── test0-0.png           # Individual generated images
        │   ├── README.md             # Configuration used
        │   ├── checkpoints/          # Finetuned model checkpoints
        │   │   ├── finetuned-model-19-012000.ckpt  # Checkpoint at epoch 19
        │   │   ├── finetuned-model-39-024000.ckpt  # Checkpoint at epoch 39
        │   │   ├── finetuned-model-59-036000.ckpt  # Checkpoint at epoch 59
        │   │   └── last.ckpt         # Most recent checkpoint
        │   └── ...
        └── run_01-01-2024-11-15-20/  # Another training run
            └── ...
```

## Key Features

### Dataset Handling
- **Per-Image Reconstruction**: Uses individual image labels (y_train.npy, y_val.npy, y_test.npy) for fine-grained reconstruction
- **Reconstruction Split Strategy**: 
  - `split="train"`: Combines train and validation data for maximum training data (recommended for reconstruction)
  - `split="test"`: Uses only test data for final evaluation
- **Automatic Label Encoding**: Handles string labels by automatically encoding them to integers while preserving original mappings
- **Flexible Data Processing**: 
  - Training data: Combines train+val, then flattens (n_images × n_repeats) for maximum training samples
  - Test data: Uses test split only, averages across repeats for clean evaluation
- **Image-Label Mapping**: Automatically maps images in `image_pool` to their corresponding per-image labels
- **Shared Label Encoder**: Ensures consistent label encoding across all splits

### Model Integration
- **Auto-Detection**: Automatically finds the best model checkpoint and configuration from `experiment_dir/pretrains/`
- **Organized Storage**: All outputs saved within the experiment directory structure
- **Pretrained LDM**: Uses pretrained latent diffusion models for high-quality image generation
- **EEG Conditioning**: Adapts the diffusion model to be conditioned on EEG signals

### Training Features
- **WandB Integration**: Automatic experiment tracking and logging
- **Flexible Configuration**: Easy parameter adjustment through command line or config files
- **Resume Capability**: Can resume training from checkpoints
- **Evaluation Metrics**: Comprehensive evaluation including MSE, PCC, SSIM, and classification accuracy

## Usage

### 1. Basic Usage with YAML Configuration

```bash
python stageB_ldm_finetune_other.py --config /path/to/your/experiment/configs/reconstruction_config.yaml
```

This will use all parameters from the YAML configuration file.

### 2. Override Specific Parameters

```bash
python stageB_ldm_finetune_other.py \
    --config /path/to/experiment/configs/reconstruction_config.yaml \
    --batch_size 4 \
    --num_epoch 100 \
    --lr 1e-4
```

Command line arguments override the corresponding values in the YAML file.

### 3. Check Configuration Examples

Run the example script to see usage patterns:

```bash
python run_example.py --examples
```

Or run with default settings:

```bash
python run_example.py
```

### 4. Test Dataset Implementation

```bash
python test_dataset_other.py
```

This will test the dataset loading and verify everything works correctly.

## Configuration

### YAML Configuration File

The training is now controlled by a YAML configuration file located at:
```
experiment_dir/configs/reconstruction_config.yaml
```

Example configuration:
```yaml
# EEG Visual Reconstruction Configuration
seed: 2022
experiment_dir: "."  # Current experiment directory
label_type: "label"  # Use per-image labels for reconstruction

# Training parameters
batch_size: 5
lr: 5.3e-5
num_epoch: 200
crop_ratio: 0.2

# Model paths (auto-detected if null)
eeg_model_ckpt_path: null
eeg_model_config_path: null

# Diffusion parameters
num_samples: 5
ddim_steps: 250
```

## Advanced Usage

### Editing Configuration

To modify training parameters, edit the YAML configuration file:

```yaml
# experiment_dir/configs/reconstruction_config.yaml

# Reduce batch size for limited GPU memory
batch_size: 2
accumulate_grad: 4  # Compensate with gradient accumulation

# Shorter training for testing
num_epoch: 50

# Higher learning rate for faster convergence
lr: 1e-4

# Specify exact model paths instead of auto-detection
eeg_model_ckpt_path: "pretrains/my_eeg_model.ckpt"
eeg_model_config_path: "pretrains/model_config.yaml"
```

### Command Line Overrides

You can override any YAML parameter via command line:

```bash
# Override multiple parameters
python stageB_ldm_finetune_other.py \
    --config /path/to/config.yaml \
    --batch_size 1 \
    --accumulate_grad 8 \
    --precision 16 \
    --num_epoch 10
```

### Resuming Training

To resume from a finetuned checkpoint:

```bash
python stageB_ldm_finetune_other.py \
    --config /path/to/config.yaml \
    --checkpoint_path /path/to/experiment/results/reconstruction/run_01-01-2024-10-30-15/checkpoints/last.ckpt
```

### Multiple Experiments

Create different config files for different experiments:

```bash
# Quick test with reduced parameters
python stageB_ldm_finetune_other.py --config configs/quick_test.yaml

# Full training with optimized parameters
python stageB_ldm_finetune_other.py --config configs/full_training.yaml

# Different label type
python stageB_ldm_finetune_other.py --config configs/class_reconstruction.yaml
```

## Output

The script will create an output directory in `experiment_dir/results/reconstruction/` with:
- **Generated Images**: 
  - `samples_train.png` - Grid of generated training samples
  - `samples_test.png` - Grid of generated test samples  
  - `test_{idx}_{label}_groundtruth.png` - Original images
  - `test_{idx}_{label}_generated_{n}.png` - Generated variants
- **Finetuned Models**:
  - `checkpoints/finetuned-model-{epoch}-{step}.ckpt` - Periodic checkpoints during training
  - `checkpoints/last.ckpt` - Final finetuned model
- **Training Logs**: 
  - `README.md` - Configuration and parameters used
  - WandB logs (if configured)

The finetuned models can be used for:
- Resuming training from a checkpoint
- Running inference with the finetuned EEG-to-image model
- Further fine-tuning on different datasets

## File Organization

All files related to an experiment are organized within the experiment directory:

- **Input Data**: `experiment_dir/data/` and `experiment_dir/image_pool/`
- **Configuration**: `experiment_dir/configs/reconstruction_config.yaml`
- **Pretrained Models**: `experiment_dir/pretrains/`
- **Training Outputs**: `experiment_dir/results/reconstruction/`
- **Logs and Checkpoints**: Saved within the timestamped run directory

This organization keeps everything related to an experiment in one place for easy management and reproducibility.

## Dependencies

Make sure you have these key dependencies installed:
- PyTorch
- PyTorch Lightning
- WandB
- NumPy
- PIL/Pillow
- einops
- scikit-learn
- torchvision
- PyYAML

## Troubleshooting

### Common Issues

1. **"Config file not found"**: Ensure the YAML config file exists in `experiment_dir/configs/`

2. **"eeg_model_ckpt_path must be provided"**: Ensure you have a pretrained EEG model in `experiment_dir/pretrains/` or specify the path in the config

3. **"No images found for label"**: Check that your `image_pool` directory contains images with names matching your per-image labels

4. **Memory errors**: Reduce batch size in the YAML config or use command line override: `--batch_size 1`

5. **Label encoding errors**: Verify that your labels are consistently formatted across train/val/test splits

### Debug Steps

1. Run `python test_dataset_other.py` to verify dataset loading
2. Run `python run_example.py --examples` to see usage examples
3. Start with a small number of epochs for testing: `--num_epoch 1`
4. Use a small batch size for debugging: `--batch_size 1`

## Integration with Original Pipeline

This implementation is designed to work alongside the existing reconstruction pipeline while using the new data format. The key differences from the original `stageB_ldm_finetune.py`:

- **Configuration**: YAML-based configuration instead of Python config classes
- **Dataset Format**: Works with individual `.npy` files instead of directory-based structure
- **Label Handling**: Automatic string label encoding for per-image reconstruction
- **Image Pool**: Uses centralized image storage instead of distributed image files
- **File Organization**: Everything organized within the experiment directory
- **Reconstruction Level**: Per-image reconstruction using individual labels instead of class-level reconstruction

## Example Workflow

1. **Prepare Data**: Ensure your experiment directory has the correct structure with per-image labels
2. **Add Pretrained Models**: Place your pretrained EEG classification model in `experiment_dir/pretrains/`
3. **Configure**: Edit `experiment_dir/configs/reconstruction_config.yaml` with your parameters
4. **Verify Setup**: Run `python run_example.py --examples`
5. **Test Dataset**: Run `python test_dataset_other.py`
6. **Start Training**: Run `python stageB_ldm_finetune_other.py --config /path/to/config.yaml`
7. **Monitor Progress**: Check WandB dashboard for training metrics
8. **Evaluate Results**: Generated images will be saved in `experiment_dir/results/reconstruction/`

This implementation provides a seamless way to use the powerful EEG-to-image reconstruction capabilities with your preprocessed dataset format, focusing on per-image reconstruction for the highest fidelity results while keeping everything organized within the experiment directory.