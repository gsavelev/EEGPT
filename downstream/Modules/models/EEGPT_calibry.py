from typing import Any
from sklearn import metrics

import torch
import pytorch_lightning as pl

from ...utils_eval import get_metrics
from .lora import add_lora_to_model, apply_lora_to_linear, make_lora_module_trainable
from .EEGPT_mcae_finetune import EEGTransformer, LinearWithConstraint, EEGPTClassifier, partial, nn


class EEGPTCalibry(pl.LightningModule, EEGPTClassifier):
    def __init__(self,
                 ch_names,
                 load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckp", 
                 use_lora=False,
                 lora_rank=8,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 max_lr=1e-3,
                 steps_per_epoch=100,
                 max_epochs=10,
                 num_classes=2,
                 use_mean_pooling=False,
                 use_chan_conv=False,
                 max_norm_chan_conv=1,
                 max_norm_head=1,
                 qkv_bias=True,
                 enc_drop_rate=0.0,
                 enc_attn_drop_rate=0.0,
                 enc_drop_path_rate=0.0,
                 rec_drop_rate=0.0,
                 rec_attn_drop_rate=0.0,
                 rec_drop_path_rate=0.0,
                 use_freeze_encoder=False,
                 use_freeze_reconstructor=False,
                 interpolate_factor=2.0,
                 desired_time_len=200*10,
                 use_avg=False,
                 use_predictor=False,
                 use_out_proj=False,
                 **kwargs
                 ):
        # Initialize the LightningModule
        pl.LightningModule.__init__(self)
        
        # Initialize basic parameters needed for EEGPTClassifier later
        self.chans_num = len(ch_names)
        self.use_lora = use_lora
        
        # Store hyperparameters
        self.save_hyperparameters()
        
        # Initialize the parent EEGPTClassifier with all parameters
        EEGPTClassifier.__init__(
            self,
            num_classes=num_classes,
            in_channels=self.chans_num,
            img_size=[self.chans_num, int(2.1*256)],
            patch_stride=32,
            use_channels_names=ch_names,
            use_mean_pooling=use_mean_pooling,
            use_chan_conv=use_chan_conv,
            max_norm_chan_conv=max_norm_chan_conv,
            max_norm_head=max_norm_head,
            qkv_bias=qkv_bias,
            enc_drop_rate=enc_drop_rate,
            enc_attn_drop_rate=enc_attn_drop_rate,
            enc_drop_path_rate=enc_drop_path_rate,
            rec_drop_rate=rec_drop_rate,
            rec_attn_drop_rate=rec_attn_drop_rate,
            rec_drop_path_rate=rec_drop_path_rate,
            use_freeze_encoder=use_freeze_encoder,
            use_freeze_reconstructor=use_freeze_reconstructor,
            interpolate_factor=interpolate_factor,
            desired_time_len=desired_time_len,
            use_avg=use_avg,
            use_predictor=use_predictor,
            use_out_proj=use_out_proj,
            **kwargs
        )
        
        # Load pretrained weights
        pretrain_ckpt = torch.load(load_path)
        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]] = v
                
        self.target_encoder.load_state_dict(target_encoder_stat)
        
        # Override with custom components
        self.chan_scale = torch.nn.Parameter(torch.ones(1, self.chans_num, 1) + 0.001*torch.rand((1, self.chans_num, 1)), requires_grad=True)
        
        # Freeze model params
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # Custom linear probes
        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(240, 2, max_norm=0.25)
        
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
        # Use the class's channels_index attribute
        x = x[:, self.channels_index, :]
        x = x * self.chan_scale

        # Use eval mode for feature extraction but LoRA still works in eval mode
        self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))

        h = z.flatten(2)
        h = self.linear_probe1(self.drop(h))
        h = h.flatten(1)
        h = self.linear_probe2(h)

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
        return pl.LightningModule.on_train_epoch_start(self)

    def on_train_epoch_end(self) -> None:
        label, y_score = [], []
        for x, y in self.running_scores["train"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        rocauc = metrics.roc_auc_score(label, y_score)
        self.log('train_rocauc', rocauc, on_epoch=True, on_step=False, sync_dist=True)
        return pl.LightningModule.on_train_epoch_end(self)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
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
        return pl.LightningModule.on_validation_epoch_start(self)

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity = False
            return pl.LightningModule.on_validation_epoch_end(self)

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

        return pl.LightningModule.on_validation_epoch_end(self)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)

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

        x, logit = self.forward(x)
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