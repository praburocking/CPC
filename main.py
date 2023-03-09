#other imports
import time
import numpy as np
from timeit import default_timer as timer
from tabulate import tabulate

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
from train import train,snapshot
from validation import validation,test_down_stream
from importlib.machinery import SourceFileLoader
from utils import calculate_stats
conf_file="new_conf.py"
conf = SourceFileLoader('', conf_file).load_module()



#specification variables
audio_window=conf.audio_window
use_cuda=conf.use_cuda
timestep=conf.timestep
batch_size=conf.batch_size
logging_dir=conf.logging_dir
epochs=conf.epochs
train_split=conf.train_split
run_name = conf.run_name
logger = setup_logs(logging_dir, run_name)
writer = SummaryWriter('runs/'+conf.run_name) 

print("The run has started with the following configurations")
temp_conf_attributes=dir(conf)
meta_data=[]
for i in temp_conf_attributes:
    if not(i.startswith('__') and i.endswith('__')):
        meta_data.append([i,str(getattr(conf,i))])
logger.info(tabulate(meta_data, headers=["param_name","param_value"]))
writer.add_text('meta_data', str(tabulate(meta_data, headers=["param_name","param_value"],tablefmt="html")))
writer.flush()

#initalizing the up stream model 
device = torch.device("cuda" if use_cuda else "cpu")
us_model=None
optimizer=None
if conf.us_model is not None:
    us_model=conf.us_model(project_no_projection=conf.timestep).to(device)

    optimizer = torch.optim.Adam(us_model.parameters(), lr=conf.lr)
    if conf.load_us_model is not None:
         checkpoint = torch.load(conf.load_us_model, map_location=lambda storage, loc: storage)
         us_model.load_state_dict(checkpoint['state_dict'])
         us_model=us_model.to(device)
         logger.info("parameters loaded for the upstream model "+str(us_model.__class__.__name__)+" from the file "+conf.load_us_model) 
    for name,param in us_model.named_parameters():
        for i in conf.named_parameters_to_ignore:
            if i in name  : 
                param.requires_grad=False
                print(name)
    us_model_params = sum(p.numel() for p in us_model.parameters() if p.requires_grad)
    logger.info('===> up stream Model summary below###\n {}\n'.format(str(us_model)))
    logger.info('===> up stream total trainable parameter: {}\n'.format(us_model_params))
else:
    logger.info("WARNING::: up stream model is NONE")

ds_model=None
# changes with the training mode in down stream
if conf.ds_model is not None:
    ds_model=conf.ds_model(linear_config=conf.ds_model_config,no_classes=conf.ds_model_no_class)
    optimizer=torch.optim.Adam(ds_model.parameters(), lr=conf.lr)
    if conf.load_ds_model is not None:
        checkpoint = torch.load(conf.load_ds_model, map_location=lambda storage, loc: storage) # load everything onto CPU
        ds_model.load_state_dict(checkpoint['state_dict'])
        logger.info("parameters loaded for the down stream model "+str(ds_model.__class__.__name__)+" from the file "+conf.load_ds_model)
    ds_model=ds_model.to(device)
    for name,param in ds_model.named_parameters():
        for i in conf.named_parameters_to_ignore:
            if i in name : 
                print(name)
                param.requires_grad=False
    ds_model_params = sum(p.numel() for p in ds_model.parameters() if p.requires_grad)
    logger.info('===> down stream Model summary below###\n {}\n'.format(str(ds_model)))
    logger.info('===> down stream Model total trainable parameter: {}\n'.format(ds_model_params))
else:
    logger.info("WARNING::: down stream model is NONE")







args={"log_interval":conf.log_interval,"logging_dir":logging_dir,"epochs":epochs,"use_gpu":use_cuda,"device":device,"lr":conf.lr}

if conf.train_ds_model and ds_model is not None:
    args["down_stream_loss_fn"]=conf.ds_model_loss_fn


global_timer = timer()

def write_model_to_tf(data_loader):

    dataiter = iter(data_loader)
    data,label=next(dataiter)
    data = data.float().to(device)
    output=writer.add_graph(us_model, data)
    print(output)

def process_training(us_model,ds_model,epochs,args,train_loader,validation_loader,optimizer,batch_size,patience_thresold):
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    patience_counter=0 
    for epoch in range(1, epochs + 1):
        epoch_timer = timer()
   
        train(args,us_model, ds_model, train_loader, optimizer, epoch, batch_size,conf.train_us_model,conf.train_ds_model)
        val_loss = validation(args, us_model,ds_model,validation_loader, batch_size,conf.validate_us_model,conf.validate_ds_model)
        writer.add_scalar("loss/train",val_loss , epoch)
        writer.flush()

        # Save
        if val_loss < best_loss: 
            best_loss = val_loss
            if conf.save_us_model and us_model is not None:
                snapshot(args["logging_dir"], run_name, { 
                    'epoch': epoch + 1,
                    'state_dict': us_model.state_dict(),
                    'validation_loss': val_loss,
                    'optimizer': optimizer.state_dict(),
                })
            if conf.save_ds_model and ds_model is not None:
                snapshot(args["logging_dir"], run_name, { 
                    'epoch': epoch + 1,
                    'state_dict': ds_model.state_dict(),
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



if conf.train:
    train_loader,validation_loader=get_dataloaders(conf)
    writer.add_text('validation_data_size',str(len(validation_loader.dataset)))
    writer.add_text('train_data_size',str(len(train_loader.dataset)))
    process_training(us_model,ds_model,epochs,args,train_loader,validation_loader,optimizer,batch_size,conf.patience_thresold)
if conf.test:
    test_loader=get_dataloaders(conf)

    predicted,target=test_down_stream(args, us_model, ds_model, test_loader, batch_size)
    clasifi_report=calculate_stats(predicted,target)
    writer.add_text('classification_report', clasifi_report)

end_global_timer = timer()
logger.info("################## Success #########################")
logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))