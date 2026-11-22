# sources: 
# https://perso.esiee.fr/~chierchg/deep-learning/tutorials/transfer/transfer-3.html
# https://www.learnpytorch.io/02_pytorch_classification/#8-putting-things-together-by-building-a-multi-class-pytorch-model


from torch import nn
from protgen.models import DecoderOnlyTransformer

class ClassifierHead(nn.Module):
    
    # input from transformer: (batch_size, seq_len, embed_dim) = hidden states
    # it takes onlz the last dimension: embed_dim
    # output: (batch_size, seq_len, num_classes) -> class logits per sample
    
    def __init__(self, input_features, output_features, hidden_units=128):
        
        """Initializes all required hyperparameters for a multi-class classification model.

        Args:
            input_features (int): Number of input features to the model.
            out_features (int): Number of output features of the model
              (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 128.
        """
        
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.GELU(), 
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)
    
    

class ClassifierGuidedTransformer(nn.Module): 
    
    def __init__(self, transformer, classifier_head):
        
        """Initialise with a pretrained transformer and a 
        classifier head. Transformer parameters are frozen.
        
        Args:
            transformer (nn.Module): Pretrained transformer model.
            classifier_head (nn.Module): Classifier head.
        """
        
        super().__init__()
        
        # change transformer settings to freeze it 
        transformer.eval()  # set transformer to evaluation mode
        for param in transformer.parameters():
            param.requires_grad = False  # freeze transformer weights
            
        self.transformer = transformer
        self.classifier_head = classifier_head
        
    # now i freeze transformer weights -> put it to eval at all times
    # need to override / disable training mode for transformer completely ?

    def forward(self, x):
        self.transformer.eval()  # make sure transformer is in evaluation mode again
        hidden_states = self.transformer(x, return_hidden_states=True)
        output = self.classifier_head(hidden_states)
        return output
    
 

    