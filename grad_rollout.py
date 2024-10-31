import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
import time
import torch.nn.functional as F


def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    attentions = attentions[::-1]
    for attention, grad in zip(attentions, gradients):
        weights = grad
        attention_heads_fused = (attention*weights).mean(axis=1) # torch.Size([1, 12, 197, 197]) torch.Size([1, 12, 197, 197])
        attention_heads_fused[attention_heads_fused < 0] = 0
        # Drop the lowest attentions, but
        # don't drop the class token
        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
        #indices = indices[indices != 0]
        flat[0, indices] = 0
        I = torch.eye(attention_heads_fused.size(-1))
        aa = (attention_heads_fused + 1.0*I)/2
        aa = aa / aa.sum(dim=-1)
        result = torch.matmul(aa, result) # torch.Size([1, 197, 197])
        break
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :] # [196]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width)
    mask = mask / mask.max()
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(224, 224), mode = 'bilinear').squeeze(1)[0]
    return mask

class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self.attention_layer_name = attention_layer_name
        self.hook_registered = False
        self.hooks = []
        self.attentions = []
        self.attention_gradients = []
        # self.max_length = num_heads
        self.pointer_a = 0
        self.pointer_ag = 0

    def __del__(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear() 
        self.attentions.clear()
        self.attention_gradients.clear()
        return

    def _register_hook(self):
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name:
                handle1 = module.register_forward_hook(self.get_attention)
                self.hooks.append(handle1)
                handle2 = module.register_full_backward_hook(self.get_attention_gradient)
                self.hooks.append(handle2)
            

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.attentions.clear()
        self.attention_gradients.clear()
        if(not self.hook_registered):
            self._register_hook()
            self.hook_registered = True
        self.model.zero_grad()
        output = self.model(input_tensor)
        if(len(output) == 2):
            output, _ = output
        category_mask = torch.zeros(output.size()).cuda()
        category_mask[:, category_index] = 1
        loss = (output * category_mask).sum()
        loss.backward(retain_graph = True)
        # print(f"time:{then - since}")
        return grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)
