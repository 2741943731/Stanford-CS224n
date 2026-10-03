import torch
import math
from torch import nn
import torch.nn.functional as F

class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, r = 8, alpha = 16, lora_dropout = 0.0, bias = True):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r if self.r > 0 else 0.0
        if r > 0:
            self.loraA = nn.Parameter(torch.zeros((r, in_features)))
            self.loraB = nn.Parameter(torch.zeros((out_features, r)))
            self.lora_dropout = nn.Dropout(p=lora_dropout)

            nn.init.kaiming_uniform_(self.loraA, a=math.sqrt(5))
            nn.init.zeros_(self.loraB)

        for name, param in self.named_parameters():
            if name not in ['loraA', 'loraB']:
                param.requires_grad = False

    def forward(self, x):
        if self.r > 0:
            lora_out = self.lora_dropout(x) @ self.loraA.T @ self.loraB.T * self.scaling
            return F.linear(x, self.weight, self.bias) + lora_out
        else:
            return F.linear(x, self.weight, self.bias)


    @classmethod
    def from_linear(cls, linear_layer, r = 8, alpha = 16, lora_dropout = 0.0):
        lora_layer = cls(linear_layer.in_features, linear_layer.out_features, r, alpha, lora_dropout, bias=linear_layer.bias is not None)
        # lora_layer.weight.data = linear_layer.weight.data.clone()
        lora_layer.weight.data.copy_(linear_layer.weight.data)
        if linear_layer.bias is not None:
            # lora_layer.bias.data = linear_layer.bias.data.clone()
            lora_layer.bias.data.copy_(linear_layer.bias.data)
        return lora_layer


def apply_lora(model, r = 8, alpha = 16, lora_dropout = 0.0, target_modules = ['query', 'value']):
    for name, module in list(model.named_modules()):
        # if isinstance(module, target_modules[0]) or isinstance(module, target_modules[1]):
        if type(module) == nn.Linear and any(target_module in name for target_module in target_modules):
            lora_module = LoRALinear.from_linear(module, r, alpha, lora_dropout)
            parent_module = model
            name_parts = name.split('.')
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name_parts[-1], lora_module)
    for name, param in model.named_parameters():
        if 'loraA' in name or 'loraB' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False