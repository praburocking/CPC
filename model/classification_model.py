


import torch.nn as nn
from .cpc_model1 import CPC_encoder_mlp,AutoRegressor


class EmotionClassifier_relu(nn.Module):
    def __init__(self,linear_config,no_classes,encoder_input_dim=1,encoder_output_dim=512,encoder_no_encoder=8,ar_input_dim=512,ar_output_dim=256,project_no_projection=16,project_input_dim=256,project_output_dim=512):
        super(EmotionClassifier_relu,self).__init__()
        # self.cpc_encoder=Encoder(input_dim=encoder_input_dim,no_encoder=encoder_no_encoder,output_dim=encoder_output_dim)
        self.cpc_encoder=CPC_encoder_mlp()
        #self.cpc_ar=AutoRegressor(input_dim=ar_input_dim,output_dim=ar_output_dim)
        linear_modules=[]
        for i in linear_config:
            linear_modules.append(nn.Linear(i["in_dim"],i["out_dim"]))
            linear_modules.append(nn.ReLU(inplace=True))
        self.linear_modules=nn.ModuleList(linear_modules)
        self.gru=nn.GRU(input_size=linear_config[-1]["out_dim"],hidden_size=32,batch_first=True)
        self.linear_end=nn.Linear(32,no_classes)
        self.activation=nn.Softmax()
     
    
    def forward(self,x): #==> [Batch,Channel,length]
        """
        forward(x)
        @parameter x of size [Batch,Channel,Length]
        @returns loss=None,predicted_y
        """
        
        x=x.transpose(1,2)#==>[Batch,Length,Channel]
        x=self.cpc_encoder(x) #==>[Batch,length,Channel/feature]
        for linear_module in self.linear_modules:
            x=linear_module(x)
        x,_=self.gru(x)
        x=x[:,-1,:]
        x=self.linear_end(x)
        return None,x




import torch.nn as nn
from .cpc_model1 import CPC_encoder_mlp,AutoRegressor


class DownStreamClassifier_cnn(nn.Module):
    def __init__(self,linear_config,no_classes,encoder_input_dim=1,encoder_output_dim=512,encoder_no_encoder=8,ar_input_dim=512,ar_output_dim=256,project_no_projection=16,project_input_dim=256,project_output_dim=512):
        super(DownStreamClassifier_cnn,self).__init__()
        # self.cpc_encoder=Encoder(input_dim=encoder_input_dim,no_encoder=encoder_no_encoder,output_dim=encoder_output_dim)
        self.cpc_encoder=CPC_encoder_mlp()
        #self.cpc_ar=AutoRegressor(input_dim=ar_input_dim,output_dim=ar_output_dim)
        self.cnn1=nn.Conv1d(in_channels=512,out_channels=256,kernel_size=3,stride=2)
        self.cnn2=nn.Conv1d(in_channels=256,out_channels=128,kernel_size=3)
        self.gru=nn.GRU(input_size=128,hidden_size=64,batch_first=True)
        self.linear1=nn.Linear(64,32)
        self.relu=nn.ReLU()
        self.linear_end=nn.Linear(32,no_classes)
     
    
    def forward(self,x): #==> [Batch,Channel,length]
        """
        forward(x)
        @parameter x of size [Batch,Channel,Length]
        @returns loss=None,predicted_y
        """
        
        x=x.transpose(1,2)#==>[Batch,Length,Channel]
        x=self.cpc_encoder(x) #==>[Batch,length,Channel/feature]
        x=x.transpose(1,2)
        x=self.cnn1(x)
        x=self.cnn2(x)
        x=x.transpose(1,2)
        x,_=self.gru(x)
        x=x[:,-1,:]
        x=self.relu(self.linear1(x))
        x=self.linear_end(x)
   
        return None,x

class DownStreamClassifier_gru(nn.Module):
    def __init__(self,linear_config,no_classes,encoder_input_dim=1,encoder_output_dim=512,encoder_no_encoder=8,ar_input_dim=512,ar_output_dim=256,project_no_projection=16,project_input_dim=256,project_output_dim=512):
        super(DownStreamClassifier_gru,self).__init__()
        # self.cpc_encoder=Encoder(input_dim=encoder_input_dim,no_encoder=encoder_no_encoder,output_dim=encoder_output_dim)
        self.cpc_encoder=CPC_encoder_mlp()
        self.cpc_ar=AutoRegressor(input_dim=ar_input_dim,output_dim=ar_output_dim)
        self.linear_1=nn.Linear(256,128)
        self.linear_2=nn.Linear(128,32)
        #self.gru=nn.GRU(input_size=128,hidden_size=32,batch_first=True)
        self.linear_end=nn.Linear(32,no_classes)
        self.activation=nn.Softmax()
        self.relu=nn.ReLU()
     
    
    def forward(self,x): #==> [Batch,Channel,length]
        """
        forward(x)
        @parameter x of size [Batch,Channel,Length]
        @returns loss=None,predicted_y
        """
        
        x=x.transpose(1,2)#==>[Batch,Length,Channel]
        x=self.cpc_encoder(x) #==>[Batch,length,Channel/feature]
        x=self.cpc_ar(x) #==>[Batch,length,Channel/feature]
      
        x=x[:,-1,:]
        x=self.relu(self.linear_1(x))
        x=self.relu(self.linear_2(x))
        x=self.linear_end(x)
   
        return None,x
