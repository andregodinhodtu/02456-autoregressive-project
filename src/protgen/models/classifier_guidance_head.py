# sources: 
# https://perso.esiee.fr/~chierchg/deep-learning/tutorials/transfer/transfer-3.html
# https://www.learnpytorch.io/02_pytorch_classification/#8-putting-things-together-by-building-a-multi-class-pytorch-model


from torch import nn

class ClassifierHead(nn.Module):
    
    # input from base_model: (batch_size, seq_len, embed_dim) = hidden states
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
    
    

class ClassifierGuidedLM(nn.Module): 
    
    def __init__(self, base_model, classifier_head):
        
        """Initialise with a pretrained base_model and a 
        classifier head. base_model parameters are frozen.
        
        Args:
            base_model (nn.Module): Pretrained base_model model.
            classifier_head (nn.Module): Classifier head.
        """
        
        super().__init__()
        
        # change base model settings to freeze it 
        base_model.eval()  # set base_model to evaluation mode
        for param in base_model.parameters():
            param.requires_grad = False  # freeze base_model weights
            
        self.base_model = base_model
        self.classifier_head = classifier_head
    
    
    def forward(self, x):
        self.base_model.eval()  # make sure base_model is in evaluation mode again
        hidden_states = self.base_model(x, return_hidden_states=True)
        output = self.classifier_head(hidden_states)
        return output
    
 

    