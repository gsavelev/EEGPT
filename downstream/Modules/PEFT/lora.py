import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=32, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.lora_dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Low-rank adaptation
        lora_output = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return lora_output * self.scaling
    
def apply_lora_to_linear(linear_layer, rank=8, alpha=32, dropout=0.1):
    """
    Apply LoRA to a linear layer by monkey patching its forward method
    """
    in_dim, out_dim = linear_layer.weight.shape[1], linear_layer.weight.shape[0]
    lora = LoRALayer(in_dim, out_dim, rank, alpha, dropout)
    
    original_forward = linear_layer.forward
    
    def forward_with_lora(x):
        original_output = original_forward(x)
        lora_output = lora(x)
        return original_output + lora_output
    
    # Replace forward method
    linear_layer.forward = forward_with_lora
    
    # Attach LoRA module to the linear layer so it's registered in the model
    linear_layer.lora = lora
    
    return linear_layer

def add_lora_to_attention(attention_layer, rank=8, alpha=32, dropout=0.1):
    """
    Add LoRA to the query, key, value projections in attention layer
    """
    # Apply LoRA to the QKV projection
    apply_lora_to_linear(attention_layer.qkv, rank, alpha, dropout)
    
    # Apply LoRA to output projection
    apply_lora_to_linear(attention_layer.proj, rank, alpha, dropout)
    
    return attention_layer

def make_lora_module_trainable(module):
    """
    Freeze all parameters except LoRA parameters
    """
    # Freeze all parameters
    for param in module.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    for name, submodule in module.named_modules():
        if hasattr(submodule, 'lora'):
            submodule.lora.lora_A.requires_grad = True
            submodule.lora.lora_B.requires_grad = True

def add_lora_to_model(model, target_modules=None, rank=8, alpha=32, dropout=0.1):
    """
    Add LoRA to all attention layers in the model
    
    Parameters:
    -----------
    model : nn.Module
        The model to add LoRA to
    target_modules : list or None
        List of module names to apply LoRA to. If None, applies to all attention blocks
    rank : int
        Rank of the LoRA decomposition
    alpha : int
        Scaling factor for LoRA
    dropout : float
        Dropout probability for LoRA layers
    """
    lora_params = []
    
    # Apply LoRA to encoder attention layers
    for name, block in model.named_modules():
        if hasattr(block, 'attn') and (target_modules is None or any(t in name for t in target_modules)):
            add_lora_to_attention(block.attn, rank, alpha, dropout)
            
            # Collect LoRA parameters
            lora_params.extend([
                block.attn.qkv.lora.lora_A,
                block.attn.qkv.lora.lora_B,
                block.attn.proj.lora.lora_A,
                block.attn.proj.lora.lora_B
            ])
    
    return model, lora_params 