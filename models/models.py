import torch
import torch.nn as nn
from torch.utils.data import dataloader

#Example Downstream MLP classifier
class MLP(nn.Module):
    def __init__(self, 
                 input_dimension:int,
                 output_dimension: int, # no. of classes
                 hidden_dimension: int,
                 n_hidden_layers: int = 1,
                 include_batchnorm: bool = False,
                 include_dropout: bool = False,
                 dropout_p: int = 0.5
                 ):
        
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension
        self.n_hidden_layers = n_hidden_layers
        self.include_batchnorm = include_batchnorm
        self.include_dropout = include_dropout

        self.input_layer = nn.Linear(in_features=input_dimension, out_features= hidden_dimension, bias=True)
        
        self.hidden_layers = [nn.Linear(in_features=hidden_dimension, out_features = hidden_dimension, bias=True)
                                for i in range(n_hidden_layers)]
        
        self.output_layer = nn.Linear(in_features=hidden_dimension, out_features = output_dimension,bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout_p) 
        self.batchnorm = nn.BatchNorm1d(hidden_dimension)
    
    def forward(self,x):
        ## input layer ##
        x = self.input_layer(x)
        
        if self.include_batchnorm:
            x = self.batchnorm(x)
        
        x = self.activation(x)

        if self.include_dropout:
            x = self.dropout(x)
        
        ## hidden layers ##
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        
            if self.include_batchnorm:
                x = self.batchnorm(x)
            
            x = self.activation(x)

            if self.include_dropout:
                x = self.dropout(x)
        
        ## output layer ##
        out = self.output_layer(x)
        return out


#speech emotion recognition classifier
class SER_Model(nn.Module):
    def __init__(self,backbone: nn.Module ,
                 head: nn.Module ,
                 freeze_backbone:bool = True):
        
        self.backbone = backbone #Ex: Pre-trained wav2vec can be pased here
        self.head = head
        self.freeze_backbone = freeze_backbone

        #Freeze backbone weights to not train backbone
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
    
    #Load trained model checkpoint
    def load(self,chkpoint_path):
        checkpoint = torch.load(chkpoint_path)
        self.load_state_dict(checkpoint["model"])

    def forward(self,x):
        embeddings = self.backbone(x)
        out = self.head(embeddings)
        return out , embeddings
