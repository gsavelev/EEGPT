import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=32, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize weights
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
    
    # Store original forward
    original_forward = linear_layer.forward
    
    # Define new forward with LoRA
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

def add_lora_to_model(model, rank=8, alpha=32, dropout=0.1):
    """
    Add LoRA to all attention layers in the model
    """
    # Add LoRA to encoder attention layers
    for block in model.target_encoder.blocks:
        add_lora_to_attention(block.attn, rank, alpha, dropout)
    
    # Add LoRA to reconstructor/predictor attention layers
    if hasattr(model, 'reconstructor'):
        for block in model.reconstructor.reconstructor_blocks:
            add_lora_to_attention(block.attn, rank, alpha, dropout)
    elif hasattr(model, 'predictor'):
        for block in model.predictor.predictor_blocks:
            add_lora_to_attention(block.attn, rank, alpha, dropout)
    
    # Add LoRA to classifier head
    apply_lora_to_linear(model.head, rank, alpha, dropout)
    
    # Make only LoRA parameters trainable
    make_lora_module_trainable(model)
    
    return model 