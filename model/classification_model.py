import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self,linear_config,no_classes):
        super(EmotionClassifier,self).__init__()
        linear_modules=[]
        for i in linear_config:
            linear_modules.append(nn.Linear(i["in_dim"],i["out_dim"]))
            linear_modules.append(nn.ReLU(inplace=True))
        self.linear_modules=nn.ModuleList(linear_modules)
        self.linear_end=nn.Linear(linear_config[-1]["out_dim"],no_classes)
        self.activation=nn.Softmax()
    def forward(self,x):
        out=self.linear_modules(x)
        out=self.linear_end(out)
        out=self.activation(out)
        return out
