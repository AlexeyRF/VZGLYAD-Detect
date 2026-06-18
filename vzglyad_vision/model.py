import torch
import torch.nn as nn
import math
from . import blocks

def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor if x > 0 else 0

def parse_model(d, ch):
    """
    Parses a dictionary architecture to build the PyTorch model dynamically.
    d: the model configuration dictionary
    ch: input channels (usually 3 for RGB)
    """
    if 'scales' in d and 'scale' in d:
        scale_key = d['scale']
        depth_multiple, width_multiple, max_channels = d['scales'][scale_key]
    else:
        depth_multiple, width_multiple, max_channels = d.get('depth_multiple', 1.0), d.get('width_multiple', 1.0), d.get('max_channels', 1024)

    nc = d['nc']
    layers, save, c2 = [], [], ch[-1]
    
    yaml_layers = d.get('backbone', []) + d.get('head', [])
    
    for i, (f, n, m_name, args) in enumerate(yaml_layers):
        m = getattr(blocks, m_name) if isinstance(m_name, str) else m_name
        for j, a in enumerate(args):
            if isinstance(a, str) and a == 'nc':
                args[j] = nc
                
        n = max(round(n * depth_multiple), 1) if n > 1 else n
        
        if m in (blocks.Conv, blocks.Bottleneck, blocks.SPPF, blocks.C2f):
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width_multiple, 8)
            args = [c1, c2, *args[1:]]
            if m in (blocks.Bottleneck, blocks.C2f):
                args.insert(2, n)
                n = 1
                
        elif m is blocks.Concat:
            c2 = sum(ch[x] for x in f)
            
        elif m is blocks.Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
                
        elif m is nn.Upsample:
            c2 = ch[f]
            
        else:
            c2 = ch[f]

        m_instance = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        m_instance.np = sum(x.numel() for x in m_instance.parameters())
        m_instance.i, m_instance.f, m_instance.type = i, f, t
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_instance)
        if i == 0:
            ch = []
        ch.append(c2)
        
    return nn.Sequential(*layers), sorted(save)

class CustomVisionModel(nn.Module):
    """
    Independent wrapper model for inference.
    """
    def __init__(self, cfg_dict, ch=3):
        super().__init__()
        self.model, self.save = parse_model(cfg_dict, ch=[ch])
        
    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x
