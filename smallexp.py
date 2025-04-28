import torch
import torch.nn.functional as F
import math 

B, T, C = 2, 5, 4  # 2 samples, 5 tokens each, 4 dim
n_head = 2

# fake embeddings
x = torch.randn(B, T, C)

# fake idx with padding (zero = padding)
idx = torch.tensor([
    [5, 3, 7, 0, 0],   # <-- 2 real tokens, 2 paddings
    [2, 1, 0, 0, 0]    # <-- 2 real tokens, 3 paddings
])

# build mask
mask = (idx != 0).unsqueeze(1).unsqueeze(2).to(torch.bool)  # (B, 1, 1, T)

# build qkv
qkv = torch.randn(B, T, C*3)
q, k, v = qkv.split(C, dim=2)
q = q.view(B, T, n_head, C // n_head).transpose(1,2)
k = k.view(B, T, n_head, C // n_head).transpose(1,2)
v = v.view(B, T, n_head, C // n_head).transpose(1,2)

# do attention
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
# man
att = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
att = att.masked_fill(torch.tril(torch.ones(T, T)).to(att.device) == 0, float('-inf')) 
att = F.softmax(att, dim=-1)
att @ v # (B, nh, T, hs)

print("Output shape:", out.shape)
