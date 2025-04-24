import os
import random
from glob import glob

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from EEGPT_calibry import EEGPTCalibry

def prepare_p300_data(dataset_fold, subject_id, samples_per_class=50, paradigm_type="erp", sampling_rate=256):
    """
    Prepare P300 dataset for a specific subject
    
    Parameters:
    -----------
    dataset_fold : str
        Path to the folder containing processed PhysioNetP300 data
    subject_id : int
        Subject ID for which to prepare adaptation data
    samples_per_class : int
        Number of samples to select per class (target/non-target)
    paradigm_type : str
        Type of paradigm ("erp", "motor_imagery", "sleep_staging", etc.)
    sampling_rate : int
        Sampling rate of the EEG data
        
    Returns:
    --------
    data_dict : dict
        Dictionary containing train, validation and test data
    """
    
    # Get list of all files for this subject
    class_0_files = glob(os.path.join(dataset_fold, '0', f'*.sub{subject_id}'))
    class_1_files = glob(os.path.join(dataset_fold, '1', f'*.sub{subject_id}'))
    
    print(f"Found {len(class_0_files)} non-target and {len(class_1_files)} target samples for subject {subject_id}")
    
    # Adjust samples_per_class if not enough data
    samples_per_class_0 = min(len(class_0_files), samples_per_class)
    samples_per_class_1 = min(len(class_1_files), samples_per_class)
    
    # Randomly select files
    random.seed(42)  # For reproducibility
    random.shuffle(class_0_files)
    random.shuffle(class_1_files)
    
    # Split into train, validation, test (60/20/20)
    train_ratio, val_ratio = 0.6, 0.2
    
    # Class 0 splits
    train_count_0 = int(samples_per_class_0 * train_ratio)
    val_count_0 = int(samples_per_class_0 * val_ratio)
    
    train_files_0 = class_0_files[:train_count_0]
    val_files_0 = class_0_files[train_count_0:train_count_0 + val_count_0]
    test_files_0 = class_0_files[train_count_0 + val_count_0:samples_per_class_0]
    
    # Class 1 splits
    train_count_1 = int(samples_per_class_1 * train_ratio)
    val_count_1 = int(samples_per_class_1 * val_ratio)
    
    train_files_1 = class_1_files[:train_count_1]
    val_files_1 = class_1_files[train_count_1:train_count_1 + val_count_1]
    test_files_1 = class_1_files[train_count_1 + val_count_1:samples_per_class_1]
    
    # Combine files
    train_files = train_files_0 + train_files_1
    val_files = val_files_0 + val_files_1
    test_files = test_files_0 + test_files_1
    
    # Create labels
    train_labels = [0] * len(train_files_0) + [1] * len(train_files_1)
    val_labels = [0] * len(val_files_0) + [1] * len(val_files_1)
    test_labels = [0] * len(test_files_0) + [1] * len(test_files_1)
    
    # Calculate maximum time length based on model patch size and paradigm
    model_patch_size = 32*2  # Assuming the standard model patch size
    max_time_length = calc_min_sample_length(model_patch_size, paradigm_type, sampling_rate)
    
    # Load data and cut to appropriate length
    def load_data_files(files):
        data = []
        for file in files:
            x = torch.load(file)
            # Add channel dimension if needed
            if len(x.shape) == 2:  # [channels, time]
                x = x.unsqueeze(0)  # [1, channels, time]
            
            # Cut sample to maximum time length
            _, C, T = x.shape
            if T > max_time_length:
                # Center the sample in the time dimension
                start_idx = (T - max_time_length) // 2
                x = x[:, :, start_idx:start_idx+max_time_length]
            
            data.append(x)
        return data
    
    train_data = load_data_files(train_files)
    val_data = load_data_files(val_files)
    test_data = load_data_files(test_files)
    
    # Stack and create tensors
    x_train = torch.stack(train_data, dim=0)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    
    x_val = torch.stack(val_data, dim=0)
    y_val = torch.tensor(val_labels, dtype=torch.long)
    
    x_test = torch.stack(test_data, dim=0)
    y_test = torch.tensor(test_labels, dtype=torch.long)
    
    print(f"Sample shapes - Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    print(f"Using maximum time length: {max_time_length} samples ({max_time_length/sampling_rate:.2f} seconds)")
    
    # Create data dictionary
    data_dict = {
        'train': {'x': x_train, 'y': y_train},
        'val': {'x': x_val, 'y': y_val},
        'test': {'x': x_test, 'y': y_test}
    }
    
    return data_dict

def train_eval_pipeline(subject_data, model, subject_id, max_epochs=10, batch_size=16, use_lora=True):
    """
    Train and evaluate a model with the provided subject data
    
    Parameters:
    -----------
    subject_data : dict
        Dictionary containing 'train', 'val', and 'test' datasets with 'x' and 'y' tensors
    model : EEGPTCalibry
        The model to train and evaluate
    subject_id : int
        Subject ID for saving model checkpoints
    max_epochs : int
        Maximum number of training epochs
    batch_size : int
        Batch size for training and evaluation
    use_lora : bool
        Whether the model is using LoRA adaptation
    
    Returns:
    --------
    test_results : dict
        Dictionary containing test results
    """
    # Create data loaders
    train_dataset = TensorDataset(subject_data['train']['x'], subject_data['train']['y'])
    val_dataset = TensorDataset(subject_data['val']['x'], subject_data['val']['y'])
    test_dataset = TensorDataset(subject_data['test']['x'], subject_data['test']['y'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup model save directory
    model_type = "lora" if use_lora else "baseline"
    checkpoint_dir = f'./checkpoints/subject_{subject_id}_{model_type}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'eegpt-{model_type}-{{epoch:02d}}-{{valid_loss:.4f}}',
        save_top_k=3,
        monitor='valid_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        patience=3,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    test_results = trainer.test(model, test_loader)
    
    # Save LoRA parameters if applicable
    if use_lora:
        lora_save_path = f'./lora_params_subject_{subject_id}.pth'
        model.save_lora_parameters(lora_save_path)
        print(f"LoRA parameters saved to {lora_save_path}")
    
    return test_results

def main():
    # Parameters
    dataset_fold = "/content/drive/MyDrive/tmp/PhysioNetP300/"
    subject_id = 2
    samples_per_class = 50
    
    # EEG channel names (adjust based on your data)
    ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
    
    # Model checkpoint path
    model_checkpoint = "/content/drive/MyDrive/models/eegpt_mcae_58chs_4s_large4E.ckp"
    
    # Prepare data
    subject_data = prepare_p300_data(dataset_fold, subject_id, samples_per_class)
    
    # Training parameters
    max_epochs = 10
    batch_size = 16
    
    # Create data loaders to calculate steps_per_epoch
    train_dataset = TensorDataset(subject_data['train']['x'], subject_data['train']['y'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    steps_per_epoch = len(train_loader)
    
    # Initialize model with LoRA
    lora_model = EEGPTCalibry(
        ch_names=ch_names,
        load_path=model_checkpoint,
        use_lora=True,  # Enable LoRA adaptation
        lora_rank=8,    
        lora_alpha=32,
        lora_dropout=0.1,
        max_lr=1e-3,
        steps_per_epoch=steps_per_epoch,
        max_epochs=max_epochs,
        channels_index=torch.arange(len(ch_names))  # Use all channels by default
    )
    
    # Train and evaluate LoRA model
    print("Training model with LoRA adaptation...")
    lora_results = train_eval_pipeline(subject_data, lora_model, subject_id, max_epochs, batch_size, use_lora=True)
    
    # Initialize baseline model (no LoRA, only channel scaling)
    baseline_model = EEGPTCalibry(
        ch_names=ch_names,
        load_path=model_checkpoint,
        use_lora=False,
        max_lr=1e-3,
        steps_per_epoch=steps_per_epoch,
        max_epochs=max_epochs,
        channels_index=torch.arange(len(ch_names))  # Use all channels by default
    )
    
    # Train and evaluate baseline model
    print("Training baseline model (no LoRA)...")
    baseline_results = train_eval_pipeline(subject_data, baseline_model, subject_id, max_epochs, batch_size, use_lora=False)
    
    # Compare results
    print("\nResults comparison:")
    print(f"LoRA model test metrics: {lora_results}")
    print(f"Baseline model test metrics: {baseline_results}")

if __name__ == "__main__":
    main() 