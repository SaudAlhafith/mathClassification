import time
import math
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import inspect
from torch.nn import functional as F

# -----------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # 50,000 BPE + 256 bytes + <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, mask=None):
        B, T, C = x.size()

        qkv = self.c_attn(x) # (B, T, C*3)
        q, k, v = qkv.split(self.n_embd, dim=2) # (B, T, C), (B, T, C), (B, T, C) splitting on dim 2
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # att = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, hs)

        # Enhancment: The 4th, speeding up from 130ms to 96ms
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig, num_classes=7):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # (B, T, vocab_size)
        self.classifier_head = nn.Linear(config.n_embd, num_classes, bias=False)
        
        # weight sharing scheme
        # self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}."

        mask = (idx != 0).unsqueeze(1).unsqueeze(2).bool() # (B, 1, 1, T)
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, C)
        tok_emb = self.transformer.wte(idx) # (B, T, C)
        x = tok_emb + pos_emb # (B, T, C)

        for block in self.transformer.h:
            x = block(x, mask) # (B, T, C)
        
        x = self.transformer.ln_f(x) # (B, T, C)
        # logits = self.lm_head(x) # (B, T, vocab_size)
        logits = self.classifier_head(x) # (B, T, num_classes)
        logits = logits[:, -1, :]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, classes_num=8):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config, num_classes=classes_num)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        sd_keys_hf.remove('lm_head.weight')
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys)-1, f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(t) for t in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # for param in model.transformer.parameters():
        #     param.requires_grad = False

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
    
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        
        if master_process:
            print(f"num deayed parameters tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num nodecayed parameters tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            print(f"using fused AdamW: {use_fused}")
            
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------
import tiktoken
import numpy as np
import pandas as pd

enc = tiktoken.get_encoding("gpt2")

def load_tokens(T, dataframe):
    
    npt = dataframe["Question"].tolist() # Already tokenized
    labels = dataframe["label"].tolist() # (B, T)

    x = torch.zeros((len(npt), T), dtype=torch.long)

    for i in range(len(npt)):
        npt[i] = npt[i][-T:] # (T)
        tokens = torch.tensor(npt[i], dtype=torch.long) # (T)
        x[i, :len(tokens)] = tokens # (B, T)

    y = torch.tensor(labels, dtype=torch.long) # (B)

    return x, y

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_train = pd.read_csv('train.csv')
        data_train["Question"] = data_train["Question"].apply(lambda x: enc.encode(x))
        n = int(data_train.shape[0] * 0.8)
        train_set = data_train[:n]
        val_set = data_train[n:]
        self.data = train_set if split == 'train' else val_set
        assert len(self.data) > 0, f"no data found for split {split}"
        if master_process:
            print(f"found {len(self.data)} samples for split {split}")

        # state, init at shard zero
        self.reset()

    def reset(self):
        self.tokens, self.labels = load_tokens(self.T, self.data) # (B, T)
        self.current_position = self.B * self.process_rank # process will at the start will be = 0

    def next_batch(self):
        B, T = self.B, self.T
        x = self.tokens[self.current_position: self.current_position + B]
        y = self.labels[self.current_position: self.current_position + B]

        self.current_position += B * self.num_processes

        if self.current_position + (B * self.num_processes) > len(self.tokens):
            self.current_position = B * self.process_rank
        return x, y

# ------------------------------------------------
# attempt to autodetect the device
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:

    assert torch.cuda.is_available(), "For now i think we need cuda for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # On single node
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # Only used on multi node 
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    print(f"I'm here {device}")
    torch.cuda.set_device(ddp_local_rank)
    master_process = ddp_rank == 0
else:
    # vanilla, non-DDP
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"using device: {device}")
          
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 512k tokens per step
B = 2 # batch size in a step
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

ixtolabel = { 
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology" 
    }
classes_num = len(ixtolabel)
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
# the dimension of the output that are not in our dataset is going to be -inf.
# Enhancment: The 1st, speeding up from 1000ms to 333ms
torch.set_float32_matmul_precision('high')
# get logits 
# Enhancment: The 5th, speeding up from 96ms to 93ms
model = GPT.from_pretrained('gpt2', classes_num=classes_num) # (B, T, vocab_size)
model.to(device)

# model.load_state_dict(torch.load('modeeel.pth', map_location=device), strict=False)
# model.eval()

# test_set = pd.read_csv('test.csv')
# test_set["Question"] = test_set["Question"].apply(lambda x: enc.encode(x))
    
# npt = test_set["Question"].tolist() # Already tokenized
# predictions = []
# with torch.no_grad():
#     for tokens in npt:
#         tokens = tokens # (T)
#         q = torch.tensor(tokens, dtype=torch.long).to(device)

#         if q.shape[0] % T != 0:
#             mult = q.shape[0] // T + 1
#             padding = torch.zeros((mult * T - q.shape[0]), dtype=torch.long).to(device)
#             q = torch.cat((q, padding), dim=0)
        
#         q = q.view(-1, T)
#         logits, _ = model(q) # (1, T, num_classes)
#         avg_logits = logits.mean(dim=1, keepdim=True)
#         predictions.append(avg_logits.argmax(dim=-1).item())  # Add .item() to get the scalar value

# test_set["label"] = predictions
# test_set["translated_label"] = test_set["label"].map(ixtolabel)

# test_set.to_csv('submissioncumi.csv', index=False)

# import sys; sys.exit(0)

# Enhancment: The 3nd, speeding up from 300ms to 130ms
# model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 1
max_steps = 100

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()

    # once in a while evaluate our validation loss
    if step % 50 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 50
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # Enhancment: The 2nd, speeding up from 333ms to 300ms
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to the SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right.
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # Specific Training Details from gpt3: clipping the global norm of the gradient at 1.0, This prevent shocking the model with very high updates. We upperbound the grads
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms, tok/sec : {tokens_per_sec:.2f}")

# store the model state dict
if master_process:
    if ddp:
        raw_model = model.module
    torch.save(raw_model.state_dict(), 'model.pth')
    print("model saved to model_more.pth")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)

