import torch
import torch.nn as nn
import math

class ModelConfig:
    def __init__(self,
                 vocab_size,
                 max_seq_length=1024,
                 d_model=512,
                 num_heads=8,
                 num_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 activation='gelu',
                 layer_norm_eps=1e-5,
                 initializer_range=0.02,
                 pad_token_id=0,
                 use_cache=True,
                 gradient_checkpointing=False):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        assert self.head_dim * config.num_heads == config.d_model, "d_model must be divisible by num_heads"
        
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None, past_key_value=None, use_cache=False):
        batch_size, seq_len, d_model = x.size()
        
        # QKV projections
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Handle past key values for inference
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        if use_cache:
            present = (k, v)
        else:
            present = None
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        
        # Attention output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        out = self.resid_dropout(self.proj(out))
        return out, present

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x, mask=None, past_key_value=None, use_cache=False):
        # Attention with residual connection
        attn_output, present = self.attention(self.norm1(x), mask, past_key_value, use_cache)
        x = x + attn_output
        
        # Feedforward with residual connection
        x = x + self.ff(self.norm2(x))
        
        if use_cache:
            return x, present
        return x, None

class SimpleLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_seq_length, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.final_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def get_input_embeddings(self):
        return self.embedding
        
    def forward(self, x, mask=None, past_key_values=None, use_cache=False):
        batch_size, seq_length = x.size()
        
        # Input embedding + positional encoding
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :seq_length]
        x = self.dropout(x)
        
        # Initialize present key values if using cache
        presents = () if use_cache else None
        
        # Transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            x, present = block(x, mask, past_key_value, use_cache)
            
            if use_cache:
                presents = presents + (present,)
        
        x = self.norm_f(x)
        logits = self.final_layer(x)
        
        if use_cache:
            return logits, presents
        return logits 