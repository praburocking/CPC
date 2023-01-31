import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self,input_dim:int,no_encoder:int=8,output_dim:int=512):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x): #X==>[Batch,length]
        x=x.unsqueeze(1) #==>[Batch,channel,length] where channel is 1
        x=self.encoder(x)
        return x   #==> [Batch,channel,length] where channel is 512

class AutoRegressor(nn.Module):
    def __init__(self,input_size=512,output_dim=256):
        self.gru=nn.GRU(input_size=input_size, hidden_size=output_dim, num_layers=1, batch_first=True, bidirectional=False)
    def forward(self,x): # ==> [Batch,Channel,Length] where channel is 512
        x=x.transpose(1,2)#==>[Batch,Length,Channel]
        x=self.gru(x) #==> [Batch,Length,Channel]
        return x #==> [Batch,Length,Channel]

class Projection(nn.Module):
    def __init__(self,no_of_projections:int=16,input_dim=256,output_dim=512):
        self.ws=nn.ModuleList([nn.Linear(input_dim,output_dim) for i in range(no_of_projections)])
    def forward(self,x):
        output=torch.empty(x.shape)
        for i,w in enumerate(self.ws):
            output[:,i,:]=w(x[:,i,:])
        return output
