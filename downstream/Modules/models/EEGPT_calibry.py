from typing import Any
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pytorch_lightning as pl

from ...utils_eval import get_metrics
from .lora import add_lora_to_model
from .EEGPT_mcae_finetune import EEGTransformer, LinearWithConstraint


class EEGPTCalibry(pl.LightningModule):
    def __init__(self,
                 ch_names,
                 load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckp", 
                 num_classes=2,
                 max_lr=1e-3,
                 steps_per_epoch=100,
                 max_epochs=10,
                 qkv_bias=True,
                 enc_drop_rate=0.0,
                 enc_attn_drop_rate=0.0,
                 enc_drop_path_rate=0.0,
                 use_lora=False,
                 lora_rank=8,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 ):
        super().__init__()
        
        self.chans_num = len(ch_names)
        self.num_classes = num_classes
        self.use_lora = use_lora
        self.lora_params = None
        
        # Store hyperparameters
        self.save_hyperparameters()
        
        # Create target encoder directly
        self.target_encoder = EEGTransformer(
            img_size=[self.chans_num, int(2.1*256)],
            patch_size=32*2,
            patch_stride=32,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=enc_drop_rate,
            attn_drop_rate=enc_attn_drop_rate,
            drop_path_rate=enc_drop_path_rate,
            init_std=0.02,
            qkv_bias=qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        
        self.chan_ids = self.target_encoder.prepare_chan_ids(ch_names)
        
        # Load pretrained weights
        pretrain_ckpt = torch.load(load_path)
        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]] = v
                
        self.target_encoder.load_state_dict(target_encoder_stat)
        
        # Custom channel scaling 
        self.chan_scale = torch.nn.Parameter(torch.ones(1, self.chans_num, 1) + 0.001*torch.rand((1, self.chans_num, 1)), requires_grad=True)
        
        # Freeze model params
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # Custom linear probes 
        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(240, self.num_classes, max_norm=0.25)
        
        # Add LoRA if requested
        if use_lora:
            self.target_encoder, self.lora_params = add_lora_to_model(
                self.target_encoder, 
                rank=lora_rank, 
                alpha=lora_alpha, 
                dropout=lora_dropout
            )
            
        self.drop = torch.nn.Dropout(p=0.50)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True
        
        # Store optimization parameters
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs

    def forward(self, x):
        x = x.to(torch.float)
        x = x - x.mean(dim=-2, keepdim=True)
        
        print(f"[DEBUG] Input shape: {x.shape}, Expected channels: {self.chans_num}") # Debug print

        # Check if channel dimensions match
        if self.chans_num != x.shape[1]:
            print(f"Warning: Input has {x.shape[1]} channels but model expects {self.chans_num}. Adjusting...")
            if x.shape[1] > self.chans_num:
                x = x[:, :self.chans_num, :] # Select first self.chans_num channels
                print(f"[DEBUG] Truncated shape: {x.shape}") # Debug print
            else:
                padding_channels = self.chans_num - x.shape[1]
                padding = torch.zeros(x.shape[0], padding_channels, x.shape[2], device=x.device, dtype=x.dtype) # Ensure dtype matches
                x = torch.cat([x, padding], dim=1) # Pad with zeros
                print(f"[DEBUG] Padded shape: {x.shape}") # Debug print
        
        # Ensure chan_ids are prepared correctly for the expected self.chans_num
        # This happens during init, so it should match self.chans_num. Add check just in case.
        if not hasattr(self, 'chan_ids') or self.chan_ids is None or len(self.chan_ids) != self.chans_num:
             print(f"Error: Mismatch between self.chans_num ({self.chans_num}) and prepared chan_ids ({len(self.chan_ids) if hasattr(self, 'chan_ids') and self.chan_ids is not None else 'None'}). Check model initialization and ch_names.")
             # Raising an error might be better than proceeding with potentially incorrect chan_ids
             raise ValueError("Channel ID mismatch detected in forward pass.")
             
        # Apply channel selection and scaling
        # Ensure chan_ids indices are valid for the current shape of x
        if x.shape[1] == self.chans_num:
            print(f"[DEBUG] Selecting channels using chan_ids (length {len(self.chan_ids)}): {self.chan_ids}") # Verbose debug print
            x_selected = x[:, self.chan_ids, :]
            print(f"[DEBUG] Shape after channel selection: {x_selected.shape}") # Verbose debug print
            
            # Ensure chan_scale matches the number of selected channels
            if self.chan_scale.shape[1] == x_selected.shape[1]:
                 x = x_selected * self.chan_scale
                 print(f"[DEBUG] Shape after scaling: {x.shape}") # Verbose debug print
            else:
                print(f"Warning: chan_scale dimension ({self.chan_scale.shape[1]}) doesn't match selected channels ({x_selected.shape[1]}). Skipping scaling.")
                x = x_selected # Skip scaling if dimensions mismatch
        else:
            # This case should not be reached if the adjustment logic above works
            print(f"Warning: Shape mismatch before channel selection. Shape is {x.shape}, expected {self.chans_num} channels. Skipping selection and scaling.")

        # Use eval mode for feature extraction but LoRA still works in eval mode (?) - check PEFT docs
        # If only training probes/LoRA, keep backbone frozen. If LoRA modifies backbone, it should be fine.
        # self.target_encoder.eval() # Commenting out - PL should handle modes. Might interfere with LoRA gradients.
        
        print(f"[DEBUG] Shape before target_encoder: {x.shape}") # Debug print
        z = self.target_encoder(x, self.chan_ids.to(x.device))
        print(f"[DEBUG] Shape after target_encoder (z): {z.shape}") # Debug print

        # Flattening logic might depend on whether a CLS token is used. Assume z is [B, N, E]
        # Original: h = z.flatten(2) # Flattens dims from 2 onwards. For [B, N, E], this is just [B, N, E].
        # The linear layer dimensions (2048, 240) suggest a specific flattening strategy was used during pretraining.
        h = z.flatten(2) 
        print(f"[DEBUG] Shape after z.flatten(2): {h.shape}") # Verbose debug print
        h = self.linear_probe1(self.drop(h))
        print(f"[DEBUG] Shape after linear_probe1: {h.shape}") # Verbose debug print
        h = h.flatten(1)
        print(f"[DEBUG] Shape after h.flatten(1): {h.shape}") # Verbose debug print
        h = self.linear_probe2(h)
        print(f"[DEBUG] Shape after linear_probe2 (logits): {h.shape}") # Verbose debug print

        # Return processed x and logits h. Returning x might be for debugging or specific loss calculation not shown.
        return x, h

    def save_lora_parameters(self, path):
        """
        Save only the LoRA parameters to a file
        """
        if not self.use_lora:
            raise ValueError("Model does not have LoRA adapters")
            
        lora_state_dict = {}
        for name, module in self.target_encoder.named_modules():
            if hasattr(module, 'lora'):
                lora_state_dict[f"{name}.lora.lora_A"] = module.lora.lora_A.data
                lora_state_dict[f"{name}.lora.lora_B"] = module.lora.lora_B.data
                
        torch.save(lora_state_dict, path)
        
    def load_lora_parameters(self, path):
        """
        Load LoRA parameters from a file
        """
        if not self.use_lora:
            raise ValueError("Model does not have LoRA adapters")
            
        lora_state_dict = torch.load(path)
        
        for name, module in self.target_encoder.named_modules():
            if hasattr(module, 'lora'):
                if f"{name}.lora.lora_A" in lora_state_dict:
                    module.lora.lora_A.data.copy_(lora_state_dict[f"{name}.lora.lora_A"])
                if f"{name}.lora.lora_B" in lora_state_dict:
                    module.lora.lora_B.data.copy_(lora_state_dict[f"{name}.lora.lora_B"])

    def on_train_epoch_start(self) -> None:
        self.running_scores["train"] = []
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        label, y_score = [], []
        for x, y in self.running_scores["train"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        rocauc = metrics.roc_auc_score(label, y_score)
        self.log('train_rocauc', rocauc, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_train_epoch_end()

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        # The forward method returns x_processed, logit
        _, logit = self.forward(x) # We only need the logits for loss calculation here
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        y_score = logit
        y_score = torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["train"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity = False
            return super().on_validation_epoch_end()

        label, y_score = [], []
        for x, y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)

        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)

        metrics_list = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics_list, True)

        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)

        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        
        # The forward method returns x_processed, logit
        _, logit = self.forward(x) # We only need the logits for loss calculation here

        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()

        loss = self.loss_fn(logit, label)
        y_score = logit
        y_score = torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        x, y = batch
        label = y.long()
        
        # The forward method returns x_processed, logit
        _, logit = self.forward(x) # We only need the logits here
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        y_score = logit
        y_score = torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["test"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        
        # Logging to TensorBoard by default
        self.log('test_loss', loss, on_epoch=True, on_step=False)
        self.log('test_acc', accuracy, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        # Parameters to optimize: channel scale and linear probes plus LoRA params if used
        params_to_optimize = [self.chan_scale] + list(self.linear_probe1.parameters()) + list(self.linear_probe2.parameters())
        
        # Add LoRA parameters if used
        if self.use_lora and self.lora_params:
            params_to_optimize.extend(self.lora_params)
        
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            weight_decay=0.01
        )

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.max_lr, 
            steps_per_epoch=self.steps_per_epoch, 
            epochs=self.max_epochs, 
            pct_start=0.2
        )
        
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'valid_loss',
            'strict': True,
            'name': None,
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_dict}