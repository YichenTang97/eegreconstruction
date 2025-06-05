import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb
import yaml
import os
from pathlib import Path
from pytorch.data_setup.DataModule_other_dataset import DataModule_Other
from pytorch.models.EEGNet import EEGNetv4
from pytorch.models.TSception import TSception
from pytorch.models.EEGChannelNet import ChannelNet
from pytorch.models.EEGNet_Embedding_version import EEGNet_Embedding
from pytorch.models.Conformer import Conformer

def read_config(config_path: str):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def merge_dicts(dict1, dict2):
    """Recursively merge dict2 into dict1."""
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dicts(dict1[key], dict2[key])
            elif key == "name" and isinstance(dict1[key], str) and isinstance(dict2[key], str):
                dict1[key] += dict2[key]
            else:
                dict1[key] = dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1

def combine_configs(head_config):
    assert "combine" in head_config, "Config is no head config!"
    head_config = head_config.pop("combine")
    for yaml_path in head_config:
        if "combined_config" in locals():
            combined_config = merge_dicts(combined_config, read_config(yaml_path))
        else:
            combined_config = read_config(yaml_path)
    return combined_config

def setup_experiment_dirs(experiment_name: str):
    """Create experiment directory structure."""
    experiment_dir = Path("experiments") / experiment_name
    results_dir = experiment_dir / "results"
    checkpoints_dir = results_dir / "checkpoints"
    wandb_logs_dir = results_dir / "wandb_logs"
    
    # Create directories if they don't exist
    for dir_path in [experiment_dir, results_dir, checkpoints_dir, wandb_logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'experiment_dir': experiment_dir,
        'results_dir': results_dir,
        'checkpoints_dir': checkpoints_dir,
        'wandb_logs_dir': wandb_logs_dir
    }

def main(config=None):
    # Initialize a new wandb run
    wandb.init(config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get experiment name from config for directory setup
    experiment_name = wandb.config.get("experiment_name", "default_experiment")
    dirs = setup_experiment_dirs(experiment_name)
    
    # Initialize data module with the new DataModule_Other
    dm = DataModule_Other(**wandb.config["datamodule"])
    
    # Print dataset information
    print("Dataset Information:")
    dataset_info = dm.get_dataset_info()
    for split, info in dataset_info.items():
        print(f"{split.upper()}: {info}")
    
    # Initialize model with correct input dimensions
    model_kwargs = wandb.config["model"].copy()
    model_kwargs["in_chans"] = dm.dims[0]  # number of channels
    model_kwargs["n_classes"] = dm.num_classes
    model_kwargs["input_window_samples"] = dm.dims[1]  # number of time points
    model_kwargs["epochs"] = wandb.config.trainer["max_epochs"]
    
    if wandb.config["model_name"] == "TSCEPTION":
        model = TSception(**model_kwargs) 
    elif wandb.config["model_name"] == "EEGNET":
        model = EEGNetv4(**model_kwargs) 
    elif wandb.config["model_name"] == "CHANNELNET":
        model = ChannelNet(**model_kwargs)
    elif wandb.config["model_name"] == "CONFORMER":
        model = Conformer(**model_kwargs)
    elif wandb.config["model_name"] == "EEGNET_Embedding":
        model = EEGNet_Embedding(**model_kwargs)
    else:
        raise ValueError(f"Unknown model name: {wandb.config['model_name']}")
    
    model = model.to(device)

    # Create a ModelCheckpoint callback
    if wandb.config["final_model"] == False:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirs['checkpoints_dir'],
            monitor="val_acc",
            mode="max",
            filename="best-model-{epoch:02d}-{val_acc:.2f}",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        )
    else:
        print("FINAL MODEL - saving weights only after last epoch")
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirs['checkpoints_dir'],
            monitor=None,
            filename="final-model",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=wandb.config.trainer["max_epochs"],
        logger=pl.loggers.WandbLogger(save_dir=str(dirs['wandb_logs_dir'])),
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=str(dirs['checkpoints_dir']), 
        accelerator="auto"
    )

    # Train model 
    trainer.fit(model=model, datamodule=dm)

    # Test model
    if wandb.config["final_model"] == True:
        trainer.test(datamodule=dm)

    wandb.run.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train models on other datasets with Wandb sweeps")
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True,
        help="Path to config file")
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment (will override config if provided)")
    args = parser.parse_args()

    # Read and combine config file
    sweep_config = read_config(config_path=args.config_path)
    print(f"Using config: {args.config_path}")
    
    if "combine" in sweep_config:
        sweep_config = combine_configs(sweep_config)
    
    # Override experiment name if provided
    if args.experiment_name:
        if "parameters" not in sweep_config:
            sweep_config["parameters"] = {}
        sweep_config["parameters"]["experiment_name"] = {"value": args.experiment_name}
    
    # Ensure experiment_name is in the config
    if "experiment_name" not in sweep_config.get("parameters", {}):
        # Extract experiment name from config file path or use default
        config_name = Path(args.config_path).stem
        sweep_config.setdefault("parameters", {})["experiment_name"] = {"value": config_name}
    
    # Initialize and run sweep
    sweep_id = wandb.sweep(sweep_config, project=sweep_config["name"])
    wandb.agent(sweep_id, function=main) 