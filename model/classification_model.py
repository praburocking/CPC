import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self,linear_config,no_classes):
        super(EmotionClassifier,self).__init__()
        linear_modules=[]
        for i in linear_config:
            linear_modules.append(nn.Linear(i["in_dim"],i["out_dim"]))
            linear_modules.append(nn.ReLU(inplace=True))
        self.linear_modules=nn.ModuleList(linear_modules)
        self.gru=nn.GRU(input_size=linear_config[-1]["out_dim"],hidden_size=32,batch_first=True)
        self.linear_end=nn.Linear(32,no_classes)
        self.activation=nn.Softmax()
    def forward(self,x):
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
