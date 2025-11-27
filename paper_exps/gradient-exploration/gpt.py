import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE (rotary) positions on Q,K."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_embd = cfg.n_embd

        self.key = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cache(self, T: int, device, dtype, start_pos: int = 0):
        # absolute positions [start_pos, start_pos+T)
        t = torch.arange(
            start_pos, start_pos + T, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("t,f->tf", t, self.inv_freq)  # [T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
        cos = emb.cos().to(dtype)[None, None, :, :]  # [1,1,T,head_dim]
        sin = emb.sin().to(dtype)[None, None, :, :]
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    @staticmethod
    def _apply_rope(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return (x * cos) + (CausalSelfAttention._rotate_half(x) * sin)

    def forward(self, x, start_pos: int = 0):
        B, T, C = x.size()

        q = (
            self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        )  # [B,h,T,hd]
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self._rope_cache(T, x.device, x.dtype, start_pos)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,h,T,T]
        # causal mask for future positions
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # [B,h,T,hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.drop(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, start_pos: int = 0):
        x = x + self.attn(self.ln1(x), start_pos=start_pos)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos: int = 0):
        B, T = idx.size()
        if T > self.cfg.block_size:
            raise ValueError("Sequence too long for block_size")

        x = self.tok_emb(idx)  # [B,T,C]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, start_pos=start_pos)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B,T,V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            start_pos = max(0, idx.size(1) - idx_cond.size(1))
            logits, _ = self(idx_cond, start_pos=start_pos)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


# example usage:
# cfg = GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768, block_size=1024)
# model = GPT(cfg)
# idx = torch.randint(0, cfg.vocab_size, (2, 16))
# logits, loss = model(idx, targets=idx)
# out = model.generate(idx[:, :1], max_new_tokens=50, temperature=0.8, top_k=50)
