from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch 

@dataclass
class ModelConfig:
    n_layer : int
    batch_size : int
    num_heads : int
    n_embd : int
    block_size : int
    head_size : int
    device : str
    vocab_size : int
    special_tokens : list[str]
    split_tokens : list[str]
    dropout : float = 0
    lr : float = 0.00001
    val_ratio : float = 0.1
    seed : int = 31337
    verbose: bool = False
    dataset_name : str = ''
    min_val_docs : int = 20


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd == config.num_heads * config.head_size
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.dropout_v = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)                           # each (B, T, C)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_v if self.training else 0.0,
        )                                                                # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)             # concatenate heads
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GptLanguageModel(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        if not self.device:
            self.device = 'cpu'

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.emb_dropout = nn.Dropout(config.dropout)

        # reinit weights
        self.apply(self._init_weights)
        residual_std = 0.02 * (2 * config.n_layer) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith('sa.proj.weight') or name.endswith('ffwd.net.2.weight'):
                nn.init.normal_(p, mean=0.0, std=residual_std)

        self.to(self.device)

        if self.config.verbose:
            param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Model created, param count: {param_count/1e6:.2f}M.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb
        x = self.emb_dropout(x) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad
    def generate(self, prompt, max_new_tokens = 512):
        if isinstance(prompt, list):
            prompt = torch.tensor(prompt, dtype=torch.long, device=self.config.device)
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        self.eval()
        generated = prompt.clone()
        for _ in range(max_new_tokens):
            prompt_crop = generated[:, -self.config.block_size:]
            logits, loss = self(prompt_crop)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            y = torch.multinomial(probs, 1)
            generated = torch.cat((generated, y), dim=1)

        if generated.size(0) == 1:
            generated = generated.squeeze(0)
        return generated

