import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

class MLPLM(nn.Module):
    """
    Multi Layer Precepton, We need to describe this better!
    """
    def __init__(self, vocab_size, context=3, stride = 1, d_emb=32, d_hidden=256, pad_id=0):

        # Init, not any init a super init
        super().__init__()

        # How much context can we get from it, and what is the padding id
        self.context, self.pad_id = context, pad_id

        # Stride for context on the sequence:
        self.stride_mlp = stride

        # Create the embeding layer where we look for it in a Lookup
        # Row for padding starts as all 0 so it is never updated! DECIDE
        self.emb = nn.Embedding(vocab_size, d_emb, padding_idx=pad_id)

        # Our Beautiful Network
        self.net = nn.Sequential(
            nn.Linear(context * d_emb, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, vocab_size),
        )

    def forward(self, seq, debug = False):

        if debug: print(seq)
        if debug: print(f"Lenght of the sequence: {len(seq)}")
            
        # Padding
        x = F.pad(seq, (self.context - 1, 0), value=self.pad_id) 
        if debug: print(x)
        if debug: print(f"Lenght of the sequence: {len(x)}")

        # Let's make windows of out context size
        # First argument is dimensions in which unfolds, I am just scared that
        # it might not be that one for batching but let's try it
        windows = x.unfold(0, self.context, self.stride_mlp) 
        if debug: print(windows)
        if debug: print(f"Lenght of the sequence: {len(windows)}")

        # Embed each token
        embedings = self.emb(windows)
        if debug: print(embedings)
        if debug: print(f"Lenght of the sequence: {embedings.shape}")

        # We flatten the emebdings so that they can go into the network
        flat_embedings = torch.flatten(embedings, start_dim=1)
        if debug: print(flat_embedings)
        if debug: print(f"Lenght of the sequence: {flat_embedings.shape}")

        return self.net(flat_embedings)

        # looking into the cross entropy loss function, logits or softmax outputs 
