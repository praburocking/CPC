import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self,input_dim:int=1,no_encoder:int=8,output_dim:int=512):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(input_dim, output_dim, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_dim, output_dim, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x): #X==>[Batch,length]
        x=x.unsqueeze(1) #==>[Batch,channel,length] where channel is 1
        x=self.encoder(x)
        return x   #==> [Batch,channel,length] where channel is 512

class AutoRegressor(nn.Module):
    def __init__(self,input_dim=512,output_dim=256):
        self.gru=nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1, batch_first=True, bidirectional=False)
    def forward(self,x): # ==> [Batch,Channel,Length] where channel is 512
        x=x.transpose(1,2)#==>[Batch,Length,Channel]
        x=self.gru(x) #==> [Batch,Length,Channel]
        return x #==> [Batch,Length,Channel]

class Projection(nn.Module):
    def __init__(self,no_of_projections:int=16,input_dim=256,output_dim=512):
        self.ws=nn.ModuleList([nn.Linear(input_dim,output_dim) for i in range(no_of_projections)])
    def forward(self,x): #==>[batch,length,feature/channel]
        output=[]
        for i,w in enumerate(self.ws):
            output.append(w(x))
        return output

class CPC(nn.Module):
    def __init__(self,encoder_input_dim=1,encoder_output_dim=512,encoder_no_encoder=8,ar_input_dim=512,ar_output_dim=256,project_no_projection=16,project_input_dim=256,project_output_dim=512):
        self.encoder=Encoder(input_dim=encoder_input_dim,no_encoder=encoder_no_encoder,output_dim=encoder_output_dim)
        self.ar=AutoRegressor(ar_input_dim=ar_input_dim,output_dim=ar_output_dim)
        self.projection=Projection(no_of_projections=project_no_projection,input_dim=project_input_dim,output_dim=project_output_dim)
        self.no_of_projects=project_no_projection
    
    def forward(self,x): #==> [Batch,length]
        x=self.encoder(x) #==>[Batch,Channel/feature,length]

        z=x.copy()
        x=self.ar(x) #==>[Batch,length,Channel/feature]
        x=self.projection(x) #==>[Batch,length,feature/channel]
        for i in range(z.shape-self.no_of_projects):
            # calculate the INFO-NCE by comparing the z and x 
            pass

