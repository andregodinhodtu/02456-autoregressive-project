import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Constants for the dataset and model
FAMILIES = ("AcyP", "HPr", "HU")
AA = "ACDEFGHIKLMNPQRSTVWY"
special_tokens = ["<PAD>", "<BOS>", "<EOS>"]
total_tokens = list(special_tokens) + list(AA)
IGNORE_INDEX = -1  # what F.cross_entropy(ignore_index=...) must be given for the generative task

class MLPLM(nn.Module):
    """
    Multi Layer Precepton, We need to describe this better!
    """
    def __init__(self, vocab_size, context=3, d_emb=32, d_hidden=256, pad_id=0):

        # Init, not any init a super init
        super().__init__()

        # How much context can we get from it, and what is the padding id
        self.context, self.pad_id = context, pad_id

        # Stride for context on the sequence:
        self.stride_mlp = 1

        # Create the embeding layer where we look for it in a Lookup
        # Row for padding starts as all 0 so it is never updated! DECIDE
        self.emb = nn.Embedding(vocab_size, d_emb, padding_idx=pad_id)

        # Our Beautiful Network
        self.net = nn.Sequential(
            nn.Linear(context * d_emb, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, vocab_size),
        )

    def forward(self, seq):
        
        # Padding
        x = F.pad(seq, (self.context - 1, 0), value=self.pad_id) 

        # Let's make windows of out context size
        windows = x.unfold(1, self.context, self.stride_mlp) 

        # Embed each token, then join each window's vectors
        embedings = self.emb(windows)                                 
        embedings = embedings.flatten(start_dim=2)    

        return self.net(embedings)

        # looking into the cross entropy loss function, logits or softmax outputs 
