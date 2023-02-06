#other imports
import time
import numpy as np
from timeit import default_timer as timer

#torch imports
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim


#local imports
from dataset import get_dataloaders
from model.cdc_model import CDCK2
from model.cpc_model1 import CPC
from model.classification_model import EmotionClassifier
from scheduledoptimizer import ScheduledOptim
from logger import setup_logs
from train import train,snapshot,train_down_stream
from validation import validation,validation_down_stream
from importlib.machinery import SourceFileLoader
conf_file="conf.py"
conf = SourceFileLoader('', conf_file).load_module()



#specification variables
audio_window=conf.audio_window
use_cuda=conf.use_cuda
timestep=conf.timestep
batch_size=conf.batch_size
audio_window=conf.audio_window
warmup_steps=conf.warmup_steps
logging_dir=conf.logging_dir
epochs=conf.epochs
train_split=conf.train_split
run_name = conf.run_name_us_model
logger = setup_logs(logging_dir, run_name) 

print("The run has started with the following configurations")
temp_conf_attributes=dir(conf)
for i in temp_conf_attributes:
    if not(i.startswith('__') and i.endswith('__')):
        print(i+"------"+str(getattr(conf,i)))
print("-----------end of configurations----------")

device = torch.device("cuda" if use_cuda else "cpu")
#us_model = CDCK2(timestep, batch_size, audio_window).to(device)
us_model=CPC()
ds_model=None
if conf.training_mode=="down_stream":
    ds_model=EmotionClassifier(linear_config=conf.emotion_classifier_linear_config,no_classes=conf.emotion_classifier_no_class)
    checkpoint = torch.load(conf.model_path, map_location=lambda storage, loc: storage) # load everything onto CPU
    us_model.load_state_dict(checkpoint['state_dict'])
    logger.info("parameters loaded for the model "+str(us_model.__class__.__name__)+" from the file "+conf.model_path)

#optimizer = ScheduledOptim(
 #       optim.Adam(
 #           filter(lambda p: p.requires_grad, us_model.parameters()), 
 #           betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
 #       warmup_steps)
lr = 1e-4
optimizer = torch.optim.Adam(us_model.parameters(), lr=lr)

model_params = sum(p.numel() for p in us_model.parameters() if p.requires_grad)
logger.info('### Model summary below###\n {}\n'.format(str(us_model)))
logger.info('===> Model total parameter: {}\n'.format(model_params))

args={"log_interval":100,"logging_dir":logging_dir,"epochs":epochs,"use_gpu":use_cuda,"device":device,"lr":lr}


train_loader,validation_loader,test_loader=get_dataloaders(conf)
global_timer = timer()

def process_training(us_model,ds_model,epochs,args,train_loader,optimizer,batch_size):
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1 
    for epoch in range(1, epochs + 1):
        epoch_timer = timer()
        if conf.training_mode=="up_stream":
            train(args, us_model, args["device"], train_loader, optimizer, epoch, batch_size)
            val_acc, val_loss = validation(args, us_model, device, validation_loader, batch_size)
        elif conf.training_mode=="down_stream":
            train_down_stream(args, us_model, ds_model,args["device"], train_loader, optimizer, epoch, batch_size)
            val_acc, val_loss = validation_down_stream(args, us_model, ds_model,device, validation_loader, batch_size)
        
        # Save
        if val_acc > best_acc: 
            if us_model is not None:
                best_acc = max(val_acc, best_acc)
                snapshot(args["logging_dir"], us_model.__class__.__name__, { 
                    'epoch': epoch + 1,
                    'validation_acc': val_acc, 
                    'state_dict': us_model.state_dict(),
                    'validation_loss': val_loss,
                    'optimizer': optimizer.state_dict(),
                })
            if ds_model is not None:
                snapshot(args["logging_dir"], ds_model.__class__.__name__, { 
                    'epoch': epoch + 1,
                    'validation_acc': val_acc, 
                    'state_dict': us_model.state_dict(),
                    'validation_loss': val_loss,
                    'optimizer': optimizer.state_dict(),
                })
            
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
           # optimizer.increase_delta()
            best_epoch = epoch + 1
        
    end_epoch_timer = timer()
    logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args["epochs"], end_epoch_timer - epoch_timer))


#for i in train_loader:

 #   print(i.shape)
 #   print(len(i))
 #   print(type(i))
 #   break
process_training(us_model,ds_model,epochs,args,train_loader,optimizer,batch_size)
## end 
end_global_timer = timer()
logger.info("################## Success #########################")
logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))