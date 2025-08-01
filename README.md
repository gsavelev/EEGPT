# EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals

This repository is the official implementation of EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals. 
![image](figures/EEGPT.jpg)

EEGPT, a novel 10-million-parameter pretrained transformer model designed for universal EEG feature extraction. In EEGPT, a mask-based dual self-supervised learning method for efficient feature extraction is designed. Compared to other mask-based self-supervised learning methods, it adds spatio-temporal representation alignment, constructing a self-supervised task on EEG representations with high SNR and rich semantic information instead of raw signals, thus avoiding poor feature quality extracted from low SNR signals.

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```


## Datasets

Follow the instructions in the [datasets/pretrain/readme.md](datasets/pretrain/readme.md) to download the pre-training EEG dataset.
Then run the following command to preprocess the data:

```bash
cd datasets/pretrain
python prepare_pretrain_dataset.py
```
Note: If the script encounters an error when running, you can try running it again.

For downstream tasks, follow the instructions in the [datasets/downstream/readme.md](datasets/downstream/readme.md) to download and preprocess the downstream EEG datasets.

## Pretrained Models

You can download pretrained models here:

- [EEG_large](https://figshare.com/s/e37df4f8a907a866df4b) (in the 'Files/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt') trained on mixed dataset (58-channels, 256Hz, 4s time length EEG) using patch size 64. 

For downstream tasks, you should place it into `checkpoint` folder as file name "checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt". To use the model, simply load the checkpoint and pass it to the `EEGPTClassifier` class in "downstream/Modules/models/EEGPT_mcae_finetune.py".

Other pretrained models:

- [BENDR](https://github.com/SPOClab-ca/BENDR) should be placed into `downstream/Modules/models/encoder.pt`.
- [BIOT](https://github.com/ycq091044/BIOT/tree/main/pretrained-models) should be placed into `downstream/Modules/BIOT/EEG-PREST-16-channels.ckpt`,`downstream/Modules/BIOT/EEG-SHHS+PREST-18-channels.ckpt`,`downstream/Modules/BIOT/EEG-six-datasets-18-channels.ckpt`.
- [LaBraM](https://github.com/935963004/LaBraM) should be placed into `downstream/Modules/LaBraM/labram-base.pth`.

## PRETRAINING TASK

To pretrain the model(s) in the paper, configure the `pretrain/configs.py` and run this command:

```bash
cd pretrain
python run_pretraining.py
```

## DOWNSTREAM TASK : TUAB and TUEV

To train the downstream task on TUAB and TUEV,
configure the `finetune_TUAB_EEGPT.sh` `finetune_TUEV_EEGPT.sh` and run this command:

```bash
cd downstream_tueg
pip install -r requirements.txt
./finetune_TUAB_EEGPT.sh
./finetune_TUEV_EEGPT.sh
```

## OTHER DOWNSTREAM TASKS

To train other downstream tasks,
configure the python scripts in the `downstream` folder and run this command:

```bash
cd downstream
python linear_probe_{model}_{dataset}.py
python finetune_{model}_{dataset}.py
```

## Calibry Branch Changes

The `calibry` branch introduces significant improvements to the EEGPT model with the following key changes:

### 🆕 New Features

#### 1. **LoRA (Low-Rank Adaptation) Implementation**
- **New file**: `downstream/Modules/PEFT/lora.py`
- Implements parameter-efficient fine-tuning using LoRA technique
- Adds low-rank adaptation layers to attention mechanisms
- Supports configurable rank, alpha, and dropout parameters
- Enables efficient adaptation of large pretrained models with minimal parameter updates

#### 2. **EEGPT Calibry Model**
- **New file**: `downstream/Modules/models/EEGPT_calibry.py`
- New `EEGPTCalibry` class extending PyTorch Lightning for calibration tasks
- Integrates LoRA adaptation with the base EEGPT transformer
- Supports both binary and multi-class classification
- Includes comprehensive evaluation metrics with confidence intervals
- Features channel scaling and convolutional preprocessing options

### 🔧 Technical Improvements

#### 3. **Enhanced Model Architecture**
- **Modified**: `downstream/Modules/models/EEGPT_mcae_finetune.py`
- Fixed patch embedding to handle squeezed input tensors
- Improved tensor shape handling in the transformer pipeline

#### 4. **Advanced Training Features**
- **Parameter-efficient fine-tuning**: Only LoRA parameters and probe layers are trainable
- **Flexible optimization**: Supports channel scaling vs. convolutional preprocessing
- **Comprehensive metrics**: Includes ROC-AUC, precision, recall, F1-score with bootstrapping confidence intervals
- **Checkpoint management**: Proper handling of LoRA parameter initialization and loading

### 🎯 Key Benefits

- **Efficiency**: LoRA reduces trainable parameters by ~95% while maintaining performance
- **Adaptability**: Easy switching between different adaptation strategies
- **Robustness**: Enhanced evaluation with confidence intervals and multiple metrics
- **Scalability**: Efficient fine-tuning for large pretrained models

### 📊 Usage

To use the calibry branch features:

```python
from downstream.Modules.models.EEGPT_calibry import EEGPTCalibry

# Initialize model with LoRA
model = EEGPTCalibry(
    ch_names=channel_names,
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1
)

# Train with parameter-efficient fine-tuning
# Only LoRA parameters and probe layers will be updated
```

The calibry branch represents a significant advancement in efficient EEG model adaptation, making it easier to fine-tune large pretrained models for specific downstream tasks while maintaining high performance and reducing computational requirements.
