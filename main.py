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
from dataset import RawDataset,get_dataloaders
from model.cdc_model import CDCK2
from scheduledoptimizer import ScheduledOptim
from logger import setup_logs
from train import train,snapshot
from validation import validation
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
run_name = conf.run_name
logger = setup_logs(logging_dir, run_name) 

device = torch.device("cuda" if use_cuda else "cpu")
model = CDCK2(timestep, batch_size, audio_window).to(device)

optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        warmup_steps)

model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info('### Model summary below###\n {}\n'.format(str(model)))
logger.info('===> Model total parameter: {}\n'.format(model_params))

args={"log_interval":100,"logging_dir":logging_dir,"epochs":epochs,"use_gpu":use_cuda,"device":device}


train_loader,validation_loader,test_loader=get_dataloaders(conf)
## Start training
global_timer = timer()

def process_training(model,epochs,args,train_loader,optimizer,batch_size):
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1 
    for epoch in range(1, epochs + 1):
        epoch_timer = timer()

        train(args, model, args["device"], train_loader, optimizer, epoch, batch_size)
        val_acc, val_loss = validation(args, model, device, validation_loader, batch_size)
        
        # Save
        if val_acc > best_acc: 
            best_acc = max(val_acc, best_acc)
            snapshot(args["logging_dir"], run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_acc, 
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1
        
    end_epoch_timer = timer()
    logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args["epochs"], end_epoch_timer - epoch_timer))
process_training(model,epochs,args,train_loader,optimizer,batch_size)
## end 
end_global_timer = timer()
logger.info("################## Success #########################")
logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))