import os, sys
import numpy as np
import torch
import argparse
import datetime
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
import glob
import yaml
from types import SimpleNamespace

from dataset_other import EEGDataset_Other, CustomLabelEncoder
from dc_ldm.ldm_for_eeg import eLDM, read_model_config
from eval_metrics import get_similarity_metric

def wandb_init(config, output_path):
    wandb.init( project='EEG_reconstruction_other',
                group="stageB_dc-ldm_other",
                anonymous="allow",
                config=config,
                reinit=True)
    create_readme(config, output_path)

def wandb_finish():
    wandb.finish()

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

def get_eval_metric(samples, avg=True):
    """Evaluate predicted images compared to ground truths"""
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    res_list = []
    
    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))     
    
    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None, 
                        n_way=50, num_trials=50, top_k=1, device='cuda')
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')
    return res_list, metric_list

def generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config):
    """Generate images after finetuning"""
    # Generate training samples
    grid, _ = generative_model.generate(eeg_latents_dataset_train, config.num_samples, 
                config.ddim_steps, config.HW, 10) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))
    wandb.log({'summary/samples_train': wandb.Image(grid_imgs)})

    # Generate test samples
    grid, samples = generative_model.generate(eeg_latents_dataset_test, config.num_samples, 
                config.ddim_steps, config.HW)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path,f'./samples_test.png'))
    
    # Save individual test images with label names
    for sp_idx, imgs in enumerate(samples):
        # Get the corresponding label for this sample
        if sp_idx < len(eeg_latents_dataset_test.labels):
            label_idx = eeg_latents_dataset_test.labels[sp_idx].item()
            
            # Convert back to original label name
            if eeg_latents_dataset_test.label_encoder is not None:
                label_name = eeg_latents_dataset_test.label_encoder.inverse_transform([label_idx])[0]
            else:
                label_name = str(label_idx)
            
            # Clean label name for filename (remove invalid characters)
            safe_label_name = "".join(c for c in label_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_label_name = safe_label_name.replace(' ', '_')
            
            # Save ground truth image (first image in the list)
            if len(imgs) > 0:
                ground_truth_img = rearrange(imgs[0], 'c h w -> h w c')
                gt_filename = f'test_{sp_idx:03d}_{safe_label_name}_groundtruth.png'
                Image.fromarray(ground_truth_img).save(os.path.join(config.output_path, gt_filename))
            
            # Save generated images (second image onward)
            for copy_idx, img in enumerate(imgs[1:]):
                img = rearrange(img, 'c h w -> h w c')
                filename = f'test_{sp_idx:03d}_{safe_label_name}_generated_{copy_idx}.png'
                Image.fromarray(img).save(os.path.join(config.output_path, filename))
        else:
            # Fallback to original naming if label access fails
            for copy_idx, img in enumerate(imgs[1:]):  # Still skip ground truth
                img = rearrange(img, 'c h w -> h w c')
                Image.fromarray(img).save(os.path.join(config.output_path, 
                                f'./test{sp_idx}-{copy_idx}.png'))

    wandb.log({f'summary/samples_test': wandb.Image(grid_imgs)})

    # Calculate evaluation metrics
    metric, metric_list = get_eval_metric(samples, avg=config.eval_avg)
    metric_dict = {f'summary/pair-wise_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
    metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    wandb.log(metric_dict)

def normalize(img):
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

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def find_best_model(experiment_dir):
    """Find the best model checkpoint and config in the experiment directory"""
    experiment_path = Path(experiment_dir)
    
    # Look for model checkpoints in pretrains subfolder first, then other locations
    checkpoint_patterns = [
        "pretrains/**/*.ckpt",  # Look in pretrains subfolder first
        "checkpoints/*.ckpt",
        "results/*/checkpoints/*.ckpt",
        "*.ckpt"
    ]
    
    best_ckpt = None
    for pattern in checkpoint_patterns:
        ckpt_files = list(experiment_path.glob(pattern))
        if ckpt_files:
            # Sort by modification time and take the most recent
            best_ckpt = sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1]
            break
    
    # Look for model config in pretrains subfolder first, then other locations
    config_patterns = [
        "pretrains/**/*.yaml",  # Look in pretrains subfolder first
        "configs/*.yaml",
        "config/*.yaml", 
        "*.yaml"
    ]
    
    best_config = None
    for pattern in config_patterns:
        config_files = list(experiment_path.glob(pattern))
        if config_files:
            # Prefer configs with 'final' or 'test' in name
            final_configs = [f for f in config_files if 'final' in f.name.lower() or 'test' in f.name.lower()]
            if final_configs:
                best_config = final_configs[0]
            else:
                best_config = config_files[0]
            break
    
    return best_ckpt, best_config

def load_model_config(config_path):
    """Load model configuration from yaml file"""
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None

def setup_datasets(config):
    """Setup train and test datasets with shared label encoder"""
    experiment_dir = Path(config.experiment_dir)
    
    # Image transformations
    # Training: include data augmentation with random cropping
    crop_pix = int(config.crop_ratio * config.img_size)
    img_transform_train = transforms.Compose([
        normalize,
        transforms.Resize((256, 256)),  # Shrink from 425x425 to 256x256
        random_crop(config.img_size - crop_pix, p=0.5),  # Data augmentation: random crop
        transforms.Resize((256, 256)),  # Resize back to target size
        channel_last
    ])
    
    # Test: no augmentation, just shrink to target size
    img_transform_test = transforms.Compose([
        normalize, 
        transforms.Resize((256, 256)),  # Shrink to target size
        channel_last
    ])
    
    # Create shared label encoder using training data (which includes train+val)
    train_dataset_temp = EEGDataset_Other(
        experiment_dir=experiment_dir,
        split="train",  # This will load train+val data combined
        label=config.label_type,
        image_transform=img_transform_train,
        preload_images=False
    )
    shared_label_encoder = train_dataset_temp.label_encoder
    
    # Create final datasets with shared encoder
    eeg_latents_dataset_train = EEGDataset_Other(
        experiment_dir=experiment_dir,
        split="train",  # Train + validation combined
        label=config.label_type,
        image_transform=img_transform_train,
        preload_images=False,
        label_encoder=shared_label_encoder
    )
    
    eeg_latents_dataset_test = EEGDataset_Other(
        experiment_dir=experiment_dir,
        split="test",  # Test only
        label=config.label_type, 
        image_transform=img_transform_test,
        preload_images=False,
        label_encoder=shared_label_encoder
    )
    
    # Print dataset info
    print("Training dataset info (train + validation combined):")
    print(eeg_latents_dataset_train.get_info())
    print("\nTest dataset info:")
    print(eeg_latents_dataset_test.get_info())
    
    return eeg_latents_dataset_train, eeg_latents_dataset_test

def main(config):
    # Project setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Validate required model paths - EEG model checkpoint is mandatory
    if config.eeg_model_ckpt_path is None:
        raise ValueError(
            "eeg_model_ckpt_path must be provided! "
            "EEG visual reconstruction requires a pretrained EEG model that was trained for classification. "
            "Please provide the path to your pretrained EEG model checkpoint."
        )
    
    if not Path(config.eeg_model_ckpt_path).exists():
        raise FileNotFoundError(
            f"EEG model checkpoint not found: {config.eeg_model_ckpt_path}. "
            "Please provide a valid path to your pretrained EEG classification model."
        )

    # Auto-detect model config if not provided (but checkpoint is required)
    if config.eeg_model_config_path is None:
        print(f"Auto-detecting model config in {config.experiment_dir}...")
        _, best_config = find_best_model(config.experiment_dir)
        
        if best_config:
            config.eeg_model_config_path = str(best_config)
            print(f"Found config: {config.eeg_model_config_path}")
            
            # Load and apply model config if available
            model_config = load_model_config(best_config)
            if model_config:
                print("Loaded model configuration:")
                print(model_config)
        else:
            print("Warning: No model config found, using default configuration")
    
    print(f"Using EEG model checkpoint: {config.eeg_model_ckpt_path}")
    if config.eeg_model_config_path:
        print(f"Using EEG model config: {config.eeg_model_config_path}")
    
    # Debug: Show resolved paths
    print(f"Resolved experiment directory: {config.experiment_dir}")
    if hasattr(config, 'eeg_model_ckpt_path') and config.eeg_model_ckpt_path:
        ckpt_path = Path(config.eeg_model_ckpt_path)
        print(f"EEG checkpoint path exists: {ckpt_path.exists()}")
        print(f"Absolute EEG checkpoint path: {ckpt_path.absolute()}")

    # Setup datasets
    eeg_latents_dataset_train, eeg_latents_dataset_test = setup_datasets(config)
    
    # Create generative model
    generative_model = eLDM(
        config.eeg_model_config_path, 
        config.eeg_model_ckpt_path,
        device=device, 
        pretrain_root=config.pretrain_gm_path, 
        logger=config.logger, 
        ddim_steps=config.ddim_steps, 
        global_pool=config.global_pool, 
        use_time_cond=config.use_time_cond
    )
    
    # Resume training if applicable
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        generative_model.model.load_state_dict(model_meta['model_state_dict'])
        print('Model resumed from checkpoint')
        
    # Create trainer
    trainer = create_trainer(
        config.num_epoch, 
        config.precision, 
        config.accumulate_grad, 
        config.logger, 
        check_val_every_n_epoch=5, 
        fast_dev_run=False,
        output_path=config.output_path
    )
    
    # Finetune the model
    generative_model.finetune(
        trainer, 
        eeg_latents_dataset_train, 
        eeg_latents_dataset_test,
        config.batch_size, 
        config.lr, 
        config.output_path, 
        config=config
    )

    # Generate images
    generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config)
    
    return

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning for Other Dataset', add_help=False)
    # Config file path (required)
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    
    # Override parameters (optional)
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--label_type', type=str, choices=['label', 'class'], help='Which labels to use')
    parser.add_argument('--crop_ratio', type=float, help='Crop ratio for data augmentation')

    # finetune parameters
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, help='Number of training epochs')
    parser.add_argument('--precision', type=int, help='Training precision')
    parser.add_argument('--accumulate_grad', type=int, help='Gradient accumulation steps')
    parser.add_argument('--global_pool', type=bool, help='Use global pooling')

    # diffusion sampling parameters
    parser.add_argument('--pretrain_gm_path', type=str, help='Path to pretrained diffusion model')
    parser.add_argument('--num_samples', type=int, help='Number of generated samples')
    parser.add_argument('--ddim_steps', type=int, help='DDIM sampling steps')
    parser.add_argument('--use_time_cond', type=bool, help='Use time conditioning')
    parser.add_argument('--eval_avg', type=bool, help='Average evaluation')

    # model paths (optional - will auto-detect if not provided)
    parser.add_argument('--eeg_model_config_path', type=str, help='Path to EEG model config')
    parser.add_argument('--eeg_model_ckpt_path', type=str, help='Path to EEG model checkpoint')
    
    # resume training
    parser.add_argument('--checkpoint_path', type=str, help='Path to training checkpoint to resume from')

    return parser

def update_config(args, config):
    """Update config with command line arguments"""
    for attr_name in dir(args):
        if not attr_name.startswith('_') and hasattr(args, attr_name):
            arg_value = getattr(args, attr_name)
            if arg_value is not None and attr_name != 'config':  # Don't override config file path
                setattr(config, attr_name, arg_value)
    return config

def create_readme(config, path):
    """Create README with config information"""
    config_dict = config.__dict__.copy()
    print(config_dict)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print("# EEG Reconstruction - Other Dataset", file=f)
        print("", file=f)
        print("## Configuration", file=f)
        for key, value in config_dict.items():
            print(f"- {key}: {value}", file=f)

def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2, logger=None, check_val_every_n_epoch=0, fast_dev_run=False, output_path=None):
    """Create PyTorch Lightning trainer with checkpointing enabled"""
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    callbacks = []
    
    # Add checkpoint callback to save the finetuned model
    if output_path and not fast_dev_run:
        from pytorch_lightning.callbacks import ModelCheckpoint
        
        # Save checkpoints in the output directory
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(output_path, 'checkpoints'),
            filename='finetuned-model-{epoch:02d}-{step:06d}',
            save_top_k=-1,  # Save all checkpoints (since monitor=None)
            save_last=True,  # Always save the last checkpoint
            monitor=None,  # Save based on step, not validation metric
            every_n_epochs=max(1, num_epoch // 10),  # Save every 10% of training
            save_on_train_epoch_end=True
        )
        callbacks.append(checkpoint_callback)
    
    return pl.Trainer(
        accelerator=acc, 
        max_epochs=num_epoch, 
        logger=logger, 
        precision=precision, 
        accumulate_grad_batches=accumulate_grad_batches,
        enable_checkpointing=True,  # Enable checkpointing to save finetuned model
        callbacks=callbacks,
        enable_model_summary=True, 
        gradient_clip_val=0.5,
        check_val_every_n_epoch=check_val_every_n_epoch, 
        fast_dev_run=fast_dev_run
    )

def load_config(config_path):
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to namespace for attribute access
    config = SimpleNamespace(**config_dict)
    
    # Resolve paths relative to config file location
    config_dir = config_path.parent
    
    # Make experiment_dir absolute if it's relative
    if hasattr(config, 'experiment_dir') and config.experiment_dir:
        if not Path(config.experiment_dir).is_absolute():
            experiment_dir = (config_dir / config.experiment_dir).resolve()  # Use resolve() to normalize
            config.experiment_dir = str(experiment_dir)
    
    # Get the experiment directory for resolving EEG model paths
    experiment_dir = Path(config.experiment_dir) if hasattr(config, 'experiment_dir') else config_dir
    
    # Resolve EEG model checkpoint path relative to experiment directory
    if hasattr(config, 'eeg_model_ckpt_path') and config.eeg_model_ckpt_path:
        if not Path(config.eeg_model_ckpt_path).is_absolute():
            ckpt_path = (experiment_dir / config.eeg_model_ckpt_path).resolve()  # Use resolve() to normalize
            config.eeg_model_ckpt_path = str(ckpt_path)
    
    # Resolve EEG model config path relative to experiment directory
    if hasattr(config, 'eeg_model_config_path') and config.eeg_model_config_path:
        if not Path(config.eeg_model_config_path).is_absolute():
            config_path_resolved = (experiment_dir / config.eeg_model_config_path).resolve()  # Use resolve() to normalize
            config.eeg_model_config_path = str(config_path_resolved)
    
    # Make pretrain_gm_path absolute if it's relative
    if hasattr(config, 'pretrain_gm_path') and config.pretrain_gm_path:
        if not Path(config.pretrain_gm_path).is_absolute():
            # If root_path is specified, use it as base
            if hasattr(config, 'root_path') and config.root_path:
                root_path = (config_dir / config.root_path).resolve()  # Use resolve() to normalize
                gm_path = (root_path / config.pretrain_gm_path).resolve()  # Use resolve() to normalize
                config.pretrain_gm_path = str(gm_path)
            else:
                gm_path = (config_dir / config.pretrain_gm_path).resolve()  # Use resolve() to normalize
                config.pretrain_gm_path = str(gm_path)
    
    return config

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = load_config(args.config)
    config = update_config(args, config)
    
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        ckp = config.checkpoint_path
        config = model_meta['config']
        config.checkpoint_path = ckp
        print('Resuming from checkpoint: {}'.format(config.checkpoint_path))

    # Create output directory within the experiment folder
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    experiment_path = Path(config.experiment_dir)
    experiment_name = experiment_path.name
    
    # Save results within the experiment folder structure
    output_path = experiment_path / 'results' / 'reconstruction' / f'run_{timestamp}'
    config.output_path = str(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment directory: {config.experiment_dir}")
    print(f"Output directory: {output_path}")
    
    # Initialize wandb and logger
    wandb_init(config, str(output_path))
    logger = WandbLogger()
    config.logger = logger
    
    # Run main training
    main(config)
    wandb_finish() 