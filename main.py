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
from torch.utils.tensorboard import SummaryWriter


#local imports
from dataset import get_dataloaders
from model.cdc_model import CDCK2
from model.cpc_model1 import CPC
from model.classification_model import EmotionClassifier
from scheduledoptimizer import ScheduledOptim
from logger import setup_logs
from train import train,snapshot,train_down_stream
from validation import validation,validation_down_stream,test_down_stream
from importlib.machinery import SourceFileLoader
from utils import calculate_stats
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
writer = SummaryWriter('runs/'+conf.run_name) 

print("The run has started with the following configurations")
temp_conf_attributes=dir(conf)
for i in temp_conf_attributes:
    if not(i.startswith('__') and i.endswith('__')):
        print(i+"------"+str(getattr(conf,i)))
print("-----------end of configurations----------")

#initalizing the up stream and down stream model 
device = torch.device("cuda" if use_cuda else "cpu")
us_model=CPC()
us_model=us_model.to(device)
ds_model=None
lr = 1e-4
optimizer = torch.optim.Adam(us_model.parameters(), lr=lr)
ds_model_params=None
# changes with the training mode in down stream
if conf.mode=="down_stream" or conf.mode=="test":
    ds_model=EmotionClassifier(linear_config=conf.emotion_classifier_linear_config,no_classes=conf.emotion_classifier_no_class)
    optimizer=torch.optim.Adam(ds_model.parameters(), lr=lr)
    checkpoint = torch.load(conf.us_model_path, map_location=lambda storage, loc: storage) # load everything onto CPU
    us_model.load_state_dict(checkpoint['state_dict'])
    logger.info("parameters loaded for the upstream model "+str(us_model.__class__.__name__)+" from the file "+conf.us_model_path)
    us_model=us_model.to(device)
    ds_model=ds_model.to(device)
    for name,param in us_model.named_parameters():
            param.requires_grad=False
    ds_model_params = sum(p.numel() for p in ds_model.parameters() if p.requires_grad)

if conf.mode=="test":
     checkpoint = torch.load(conf.ds_model_path, map_location=lambda storage, loc: storage) # load everything onto CPU
     ds_model.load_state_dict(checkpoint['state_dict'])
     ds_model=ds_model.to(device)
     logger.info("parameters loaded for the downstream model "+str(ds_model.__class__.__name__)+" from the file "+conf.ds_model_path)

us_model_params = sum(p.numel() for p in us_model.parameters() if p.requires_grad)

for name,param in us_model.named_parameters():
            if param.requires_grad:
                print(name)
logger.info('### up stream Model summary below###\n {}\n'.format(str(us_model)))
logger.info('===> up stream total parameter: {}\n'.format(us_model_params))

logger.info('### down stream Model summary below###\n {}\n'.format(str(ds_model)))
logger.info('===> down stream Model total parameter: {}\n'.format(ds_model_params))

args={"log_interval":10,"logging_dir":logging_dir,"epochs":epochs,"use_gpu":use_cuda,"device":device,"lr":lr,"down_stream_loss_fn":conf.down_stream_loss_fn}


#get the dataloaders
train_loader,validation_loader,test_loader=get_dataloaders(conf)
global_timer = timer()

def write_model_to_tf(data_loader):

    dataiter = iter(data_loader)
    data,label=next(dataiter)
    data = data.float().to(device)
    output=writer.add_graph(us_model, data)
    print(output)

def process_training(us_model,ds_model,epochs,args,train_loader,optimizer,batch_size,patience_thresold):
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    patience_counter=0 
    for epoch in range(1, epochs + 1):
        epoch_timer = timer()
        if conf.mode=="up_stream":
            train(args, us_model, train_loader, optimizer, epoch, batch_size)
            val_acc, val_loss = validation(args, us_model,validation_loader, batch_size)
            writer.add_scalar("up_stream_Loss/train",val_loss , epoch)
        elif conf.mode=="down_stream":
            train_down_stream(args, us_model, ds_model, train_loader, optimizer, epoch, batch_size)
            val_acc,val_loss = validation_down_stream(args, us_model, ds_model, validation_loader, batch_size)
            writer.add_scalar("down_stream_Loss/train",val_loss , epoch) 

        
        writer.flush()
        # Save
        if val_loss < best_loss: 
            model_name=conf.run_name_us_model if conf.mode=="up_stream" else conf.run_name_ds_model
            model_dict=us_model.state_dict() if conf.mode=="up_stream" else ds_model.state_dict()
            best_loss = min(val_loss, best_loss)
            snapshot(args["logging_dir"], model_name, { 
                'epoch': epoch + 1,
                'validation_acc': val_acc, 
                'state_dict': model_dict,
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
        
            best_epoch = epoch + 1
            patience_counter=0
        elif patience_counter>=patience_thresold:
            logger.info("maximum patience reached with no improvement breaking the training at epoch "+str(epoch+1) )
            break
        else:
            patience_counter=patience_counter+1 
        
    end_epoch_timer = timer()
    logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args["epochs"], end_epoch_timer - epoch_timer))


train_loader,validation_loader,test_loader=get_dataloaders(conf)
if conf.mode=="up_stream" or conf.mode=="down_stream":
    process_training(us_model,ds_model,epochs,args,train_loader,optimizer,batch_size,conf.patience_thresold)
if conf.mode=="test":
    #write_model_to_tf(test_loader)
    predicted,target=test_down_stream(args, us_model, ds_model, test_loader, batch_size)
    clasifi_report=calculate_stats(predicted,target)
    writer.add_text('classification_report', clasifi_report)

end_global_timer = timer()
logger.info("################## Success #########################")
logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))