


import torch.nn as nn
from cpc_model1 import CPC_encoder_mlp,AutoRegressor,Projection,


class EmotionClassifier(nn.Module):
    def __init__(self,linear_config,no_classes,encoder_input_dim=1,encoder_output_dim=512,encoder_no_encoder=8,ar_input_dim=512,ar_output_dim=256,project_no_projection=16,project_input_dim=256,project_output_dim=512):
        super(EmotionClassifier,self).__init__()
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
        x=x.transpose(1,2)#==>[Batch,Length,Channel]
        x=self.cpc_encoder(x) #==>[Batch,length,Channel/feature]
        #print("after encoder done "+str(x.shape))
        for linear_module in self.linear_modules:
            x=linear_module(x)
        # print("size of x before gru"+str(x.shape))
        x,_=self.gru(x)
        x=x[:,-1,:]
        # print("size of x after gru"+str(x.shape))
        x=self.linear_end(x)
        # print("size of x after linear"+str(x.shape))
        #x=self.activation(x)
        return x
