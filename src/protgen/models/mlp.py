import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, vocab_size, context=3, d_emb=32, d_hidden=256, bos_id=1):

        # Init, not any init a super init
        super().__init__()

        # How much context can we get from it, and what is the begginig of sequence id?
        self.context, self.bos_id = context, bos_id

        # Create the embeding layer where we look for it in a Lookup
        self.emb = nn.Embedding(vocab_size, d_emb)
        
        self.net = nn.Sequential(nn.Linear(context * d_emb, d_hidden), 
                                 nn.ReLU(),
                                 nn.Linear(d_hidden, vocab_size))

    def _handle_data(self,Protein_Dataset):

        pass

    def train():

        _handle_data
        pass

    def testing():

        _handle_data
        pass

