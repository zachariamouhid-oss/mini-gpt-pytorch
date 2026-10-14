import torch
import torch.nn as nn
import math
import json

def _apply_rotations(x: torch.Tensor, cos: torch.Tensor, sin:torch.Tensor):

    B, seq_len, n_heads, head_dim = x.shape
    
    first_half = x[..., : head_dim //2]
    second_half = x[..., head_dim//2 :]
    rotated_x = torch.cat([ -second_half, first_half], -1)

    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)


    x = x*cos + rotated_x*sin

    return x

def compute_loss(logits, target):
    B, seq_len, vocab_size = logits.shape
    logits = torch.reshape(logits, [B*seq_len, vocab_size])
    target = torch.reshape(target, [B*seq_len])
    return torch.nn.functional.cross_entropy(logits, target) # Return the loss function to begin backpropagation


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, n_heads, n_layers):
        super().__init__()

        self.vocab_size = vocab_size 
        self.d_model = d_model 
        self.max_seq_len = max_seq_len
        self.token_embedding_table = nn.Embedding(vocab_size, d_model) # Embedding matrix
        self.n_heads = n_heads 
        self.head_dim = d_model // n_heads 
        self.n_layers = n_layers 

        assert d_model % n_heads == 0, "d_model must be a multiple of n_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        # N transformers layer
        self.Transformers_layer = nn.ModuleList([TransformerBlock(self.d_model, self.max_seq_len, self.n_heads) for i in range(n_layers)])

        # Normalisation et last projection matrix in the vocabulary
        self.last_RSMNorm = nn.RMSNorm(d_model, eps=1e-6)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

        # Initialization of every weight
        self.apply(self._init_weight)

        # Cos and sin buffered
        cos,sin = self._precompute_rotations(max_seq_len, self.head_dim)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def _precompute_rotations(self, max_seq_len:int , head_dim: int, theta = 10000):

        tab_i = torch.arange(0, head_dim, 2).float() / head_dim
        theta_tab = theta ** (-tab_i)


        pos = torch.arange(max_seq_len)

        angles = torch.outer(pos, theta_tab)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        cos = torch.cat([cos, cos], -1)
        sin = torch.cat([sin, sin], -1)
        return (cos, sin)

    def forward(self, idx : torch.Tensor):
        # Sequence recovery
        x = self.token_embedding_table(idx)
        seq_len = x.shape[1]

        # forward through all layers
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]
        for transformer in self.Transformers_layer:
            x = transformer(x, cos, sin)

        # Normalization and last projection into a vocabulary sized tensor
        x = self.last_RSMNorm(x)
        logits = self.lm_head(x)

        return logits

    def _init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1 / math.sqrt(self.d_model))
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            torch.nn.init.ones_(module.weight)

    def generate(self, idx, max_new_tokens, temperature=1.0, eos_id=None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if idx.shape[1] > self.max_seq_len:
                    truncated_idx = idx[:, -self.max_seq_len:]
                else:
                    truncated_idx = idx

                logits = self(truncated_idx)

                next_logits = logits[:, -1, :]/temperature
                next_tokens_probas = torch.softmax(next_logits, dim = -1)
                idx_next = torch.multinomial(next_tokens_probas, num_samples=1)

                idx = torch.cat([idx, idx_next], dim = 1)

                if idx.shape[0] == 1 and eos_id is not None and (idx_next == eos_id).item():
                    break

        return idx
            
    @classmethod
    def import_model(cls, config_path, weight_path, device="cpu"):
        with open(config_path, "r") as f:
            config = json.load(f)

        model = cls(**config)
        
        checkpoint = torch.load(weight_path, map_location=device, weights_only=True)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

        return model
        

class Attention(nn.Module):
    def __init__(self, d_model, max_seq_len, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads  

        # Tensors Q, K, V are stored in the same tensor to limit memory usage, and then splitted when needed
        self.W_qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        

        self.register_buffer("mask", self._create_mask(max_seq_len))

    def _create_mask(self, max_seq_len):
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)  # 0 on the main diagonal and under, -inf above
        return mask

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, seq_len, d_model = x.shape

        # Projection of x in q, k, v
        qkv = self.W_qkv(x)

        q,k,v = torch.split(qkv, self.d_model, -1)

        q = q.view(B, seq_len, self.n_heads, self.d_head)
        k = k.view(B, seq_len, self.n_heads, self.d_head)
        v = v.view(B, seq_len, self.n_heads, self.d_head)


        # RoPE
        q = _apply_rotations(q, cos, sin)
        k = _apply_rotations(k, cos, sin)

        # Dimensions rotation --> IN pytorch, always place computable dimensions in last positions --> (B, n_heads, seq_len, d_head)
        q = torch.permute(q, (0, 2, 1, 3))
        k = torch.permute(k, (0, 2, 1, 3))
        v = torch.permute(v, (0, 2, 1, 3))  
        # Innerproduct, normalization, masking and softmax
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + self.mask[:seq_len, :seq_len]
        scores = torch.softmax(scores, dim=-1)

        out = scores @ v

        out = torch.permute(out, (0, 2, 1, 3)).reshape(B, seq_len, self.d_model)

        return self.W_out(out)
        


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.Wup = nn.Linear(d_model, 4*d_model, bias=False)
        self.Wdown = nn.Linear(4*d_model, d_model, bias=False)
        self.SiLU = nn.SiLU()

    def forward(self, x):
        # In the feed-forward, we project x in an higher dimensionnal space, where it allows deeper capture of meaning.
        # Then, after activation function, we go back to initial dimension d_model
        x=self.Wup(x)
        x=self.SiLU(x)
        return self.Wdown(x)

        

class TransformerBlock(nn.Module):
    def __init__(self, d_model, max_seq_len, n_heads):
        super().__init__()
        self.Attention = Attention(d_model, max_seq_len, n_heads)
        self.FFW = FeedForward(d_model)
        self.norm1 = nn.RMSNorm(d_model, eps=1e-6)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-6)

    def forward(self, x, cos, sin):
        # x = x + Attention(Norm1(x))
        x_norm1 = self.norm1(x)
        x = x + self.Attention(x_norm1, cos, sin)
        # x = x + FFW(Norm2(x))
        x_norm2 = self.norm2(x)
        x = x + self.FFW(x_norm2)

        return x

        
