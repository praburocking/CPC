import torch.nn as nn
import torch
from torch import matmul, diag
from torch.nn import Module, LogSoftmax
import sys
import numpy as np

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
        #print("xshape in encoder "+str(x.shape))
        x=x.unsqueeze(1) #==>[Batch,channel,length] where channel is 1
        x=self.encoder(x)
        return x   #==> [Batch,channel,length] where channel is 512

class AutoRegressor(nn.Module):
    def __init__(self,input_dim=512,output_dim=256):
        super(AutoRegressor,self).__init__()
        self.gru=nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1, batch_first=True, bidirectional=False)
    def forward(self,x): # ==> [Batch,Channel,Length] where channel is 512
        x=x.transpose(1,2)#==>[Batch,Length,Channel]
        x,_=self.gru(x) #==> [Batch,Length,Channel]
        return x #==> [Batch,Length,Channel]

class Projection(nn.Module):
    def __init__(self,no_of_projections:int=16,input_dim=256,output_dim=512):
        super(Projection,self).__init__()
        self.ws=nn.ModuleList([nn.Linear(input_dim,output_dim) for i in range(no_of_projections)])
    def forward(self,x): #==>[batch,length,feature/channel]
        predicted_future_Z=[]
        for i,cur_ws in enumerate(self.ws):
            predicted_future_Z.append(cur_ws(x))
        predicted_future_Z = torch.stack(predicted_future_Z, dim=1)
        return predicted_future_Z


class InfoNCE_loss_no_classes(Module):
    """
    The CPC loss, implemented in a way that for each batch, the other samples in the same batch
    act as the negative samples. Note that in the loss calculation, a log density ratio log(f_k)
    is used as log(f_k) = matmul(Z_future_timesteps[k-1], predicted_future_Z[i].transpose(0,1)),
    w.r.t. Eq. (3) in the original CPC paper where f_k = exp(z_{t+k}^T * W_k * c_t).
    
    _____________________________________________________________________________________________
    Input parameters:
    
    future_predicted_timesteps: The future predicted timesteps (integer or a list of integers)
        
    Z_future_timesteps: The encodings of the future timesteps, i.e. z_{t+k} where k in
                        [1, 2, ..., num_future_predicted_timesteps].
    
    predicted_future_Z: The predicted future embeddings z_{t+k} where k in
                        [1, 2, ..., num_future_predicted_timesteps]
    
    _____________________________________________________________________________________________
    
    """
    
    def __init__(self, future_predicted_timesteps=16):
        super().__init__()
        
        # We first determine whether our future_predicted_timesteps is a number or a list of numbers.
        if isinstance(future_predicted_timesteps, int):
            # future_predicted_timesteps is a number, so we have future_predicted_timesteps loss calculations
            self.future_predicted_timesteps = np.arange(1, future_predicted_timesteps + 1)
            
        elif isinstance(future_predicted_timesteps, list):
            # future_predicted_timesteps is a list of numbers, so we have len(future_predicted_timesteps) loss calculations
            self.future_predicted_timesteps = future_predicted_timesteps
            
        else:
            sys.exit('Configuration setting "future_predicted_timesteps" must be either an integer or a list of integers!')

    def forward(self, Z_future_timesteps, predicted_future_Z):
        
        log_smax = LogSoftmax(dim=1)
        
        loss = 0
        num_future_predicted_timesteps = len(self.future_predicted_timesteps)
        batch_size = Z_future_timesteps.size()[1]
        
        # We go through each future timestep and compute the loss. Z_future_timesteps is of size
        # [future_predicted_timesteps, batch_size, num_features] or [len(future_predicted_timesteps), batch_size, num_features]
        # where num_features is the size of the encoding for each timestep produced by the encoder
        # --> with default values Z_future_timesteps.size() = torch.Size([12, 8, 512])
        i = 0
        for k in self.future_predicted_timesteps:
            loss -= diag(log_smax(matmul(Z_future_timesteps[k-1], predicted_future_Z[i].transpose(0,1)))).sum(dim=0)
            i += 1
            
        loss = loss / (batch_size * num_future_predicted_timesteps)
        
        return loss

class CPC(nn.Module):
    def __init__(self,encoder_input_dim=1,encoder_output_dim=512,encoder_no_encoder=8,ar_input_dim=512,ar_output_dim=256,project_no_projection=16,project_input_dim=256,project_output_dim=512):
        super(CPC,self).__init__()
        self.encoder=Encoder(input_dim=encoder_input_dim,no_encoder=encoder_no_encoder,output_dim=encoder_output_dim)
        self.ar=AutoRegressor(input_dim=ar_input_dim,output_dim=ar_output_dim)
        self.projection=Projection(no_of_projections=project_no_projection,input_dim=project_input_dim,output_dim=project_output_dim)
        self.no_of_projects=project_no_projection
        self.infoNCELoss=InfoNCE_loss_no_classes(future_predicted_timesteps=project_no_projection)
    
    def forward(self,x): #==> [Batch,length]
        x=self.encoder(x) #==>[Batch,Channel/feature,length]
        #print("after encoder done "+str(x.shape))
        z=torch.clone(x)
        x=self.ar(x) #==>[Batch,length,Channel/feature]
        output=[]
        z=z.transpose(1,2)#===>[Batch,length,Channel/feature]
        total_loss=0
        for t in range(0,x.shape[1]-self.no_of_projects):
            cur_x=x[:,t,:]
            cur_z=z[:,t+1:(t+self.no_of_projects+1),:]
            #print("z per batch ..."+str(cur_z.shape))
            cur_x=self.projection(cur_x)
            output.append(cur_x)
            loss=self.infoNCELoss(cur_z.transpose(0,1),cur_x.transpose(0,1))
            total_loss=total_loss+loss
        
         #   print("size of cur_x "+str(cur_x.shape)+" :: size of cur_z "+str(cur_z.shape))
        #print("loss ...."+str(total_loss))
            # calculate the INFO-NCE by comparing the z and x 

        return loss,z


