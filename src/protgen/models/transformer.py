import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn



import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

# adapted from https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#causal-self-attention
class CausalMultiHeadSelfAttention(nn.Module):
    
    """ Multi-Head Self Attention module with causal masking. 
    
    It uses a combined mask of a triangular causal mask and a padding mask to both ensure that 
    only past tokens can be seen and handle sequences of varying length.
    
    Args:
        num_heads (int): Number of attention heads.
        embed_dimension (int): Dimension of the input embeddings.
        bias (bool) (default=False): Option to include bias terms in the linear layers. 
        dropout(float, default=0.1): Dropout rate for regularization.
    """ 
    
    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, dropout:float=0.1):
        
        super().__init__()
        assert embed_dimension % num_heads == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        self.head_dim = embed_dimension // num_heads
        
    

    def forward(self, x, attn_mask):
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)
        batch_size = query_projected.size(0)
        
        # separate query, key, value and reshape for multi-head attention
        query, key, value = query_projected.chunk(3, -1)
        # reshape to (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)       

        y = F.scaled_dot_product_attention(query, key, value, 
                                           attn_mask=attn_mask, 
                                           dropout_p=self.dropout, 
                                           is_causal=False)
        
        y = y.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y



# adapted from https://www.geeksforgeeks.org/deep-learning/transformer-using-pytorch/ 
# and https://www.intoai.pub/p/build-a-decoder-only-transformer
# from the paper: d_ff = 4 * d_model
class PositionWiseFeedForward(nn.Module):
    
    """ Position-wise Feed Forward Network module"""
    
    def __init__(self, d_model, d_ff, dropout=0.01):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU() # used now more
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DecoderLayer(nn.Module):
    
    """ Single Decoder Layer module"""
    
    # layernorm
    # attention
    # dropout 
    
    # layernorm
    # feedforward
    # dropout
    
    def __init__(self, embedding_dim: int, ff_dim: int, num_heads: int, dropout=0.1):
        super().__init__()
        

        self.attention = CausalMultiHeadSelfAttention(num_heads=num_heads, 
                                                      embed_dimension=embedding_dim,  
                                                      dropout=dropout)

        self.ffn = PositionWiseFeedForward(embedding_dim, ff_dim, dropout)

        # pre-LN
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        # dropout applied to sublayer outputs for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        
        # attention sublayer
        ln1_output = self.ln1(x)
        attention_output = self.attention(ln1_output, attn_mask)
        dropout_output = self.dropout(attention_output)
        x = x + dropout_output  # residual connection
        
        # ffnn sublayer
        ln2_output = self.ln2(x)
        ffn_output = self.ffn(ln2_output)
        dropout_ffn_output = self.dropout(ffn_output)
        x = x + dropout_ffn_output # residual connection

        return x




# code again adapted from the awesome https://www.intoai.pub/p/build-a-decoder-only-transformer
class DecoderOnlyTransformer(nn.Module):
    
    """ Decoder-only Transformer model for autoregressive sequence generating"""
    
    def __init__(self, vocab_size, embedding_dim, num_heads, ff_dim, 
                 num_layers, max_seq_length, pad_idx=0, dropout=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.pad_idx = pad_idx # for the padding mask that has to be made before embedding
        
        # token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # positional embeddings
        self.positional_embedding = nn.Embedding(max_seq_length, 
        embedding_dim)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # stack of Decoder blocks (DecoderLayer)
        self.decoders = nn.ModuleList([
            DecoderLayer(embedding_dim, ff_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # FINAL layernorm before output
        self.final_ln = nn.LayerNorm(embedding_dim)

        # output projection to vocabulary
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x represents token indices
        batch_size, seq_length = x.shape 
        
        # padding mask for attention
        padding_mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # causal mask for attention
        causal_mask = torch.tril(torch.ones(seq_length, seq_length, device= x.device)).unsqueeze(0).unsqueeze(1).bool()
        # combine masks
        attn_mask = causal_mask & padding_mask  # shape is now (batch_size, 1, seq_len, seq_len)
        

        # positional indices
        positions = torch.arange(0, seq_length, device = x.device).unsqueeze(0)

        # token embedding
        token_embedding = self.token_embedding(x)

        # positional embedding
        positional_embedding = self.positional_embedding(positions)

        # combine embeddings and add Dropout
        x = self.dropout(token_embedding + positional_embedding)

        # forward pass through decoder blocks
        for decoder in self.decoders:
          x = decoder(x, attn_mask=attn_mask)  # pass the attention mask to each decoder layer

        # apply LayerNorm to the output
        x = self.final_ln(x)

        # output projection to vocabulary to get logits
        logits = self.output_proj(x) 

        return logits 
