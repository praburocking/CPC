#other imports
import time
import numpy as np
from timeit import default_timer as timer
from tabulate import tabulate
import copy
import onnx
import sys
import os

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
from scheduledoptimizer import ScheduledOptim
from logger import setup_logs
from train import train,snapshot
from validation import validation,test_down_stream
from importlib.machinery import SourceFileLoader
from utils import calculate_stats,create_experiment_folder

print(sys.argv)
if len(sys.argv)<3:
    print("*** mandatory arguments not found *** ")
    print("argument 1 ---- name for the run \narugment 2 --- brief description of the run")
    sys.exit(-1)

conf_file="/scratch/kcprmo/cpc/CPC/new_conf.py"
if len(sys.argv)==4:
    conf_file=sys.argv[3]
conf = SourceFileLoader('', conf_file).load_module()
list_of_experiments=[
                    "upstream_56705mins_0.0001lr",
                    "upstream_56705mins_0.001lr",
                    "downstream_56705mins_0.001lr_libri_FESC_128AW",
                    "downstream_56705mins_0.001lr_puhelahjat_FESC_128AW",
                    "downstream_56705mins_0.0001lr_libri_FESC_128AW",
                    "downstream_56705mins_0.0001lr_puhelahjat_FESC_128AW",
                    "downstream_56705mins_0.001lr_libri_RAVDESS_128AW",
                    "downstream_56705mins_0.001lr_puhelahjat_RAVDESS_128AW",
                    "downstream_56705mins_0.0001lr_libri_RAVDESS_128AW",
                    "downstream_56705mins_0.0001lr_puhelahjat_RAVDESS_128AW",
                     "downstream_56705mins_0.001lr_libri_TIMIT_128AW",
                    "downstream_56705mins_0.001lr_puhelahjat_TIMIT_128AW",
                    "downstream_56705mins_0.0001lr_libri_TIMIT_128AW",
                    "downstream_56705mins_0.0001lr_puhelahjat_TIMIT_128AW",
                    "downstream_56705mins_0.001lr_libri_FINNISH_DIALECT_128AW",
                    "downstream_56705mins_0.001lr_puhelahjat_FINNISH_DIALECT_128AW",
                    "downstream_56705mins_0.0001lr_libri_FINNISH_DIALECT_128AW",
                    "downstream_56705mins_0.0001lr_puhelahjat_FINNISH_DIALECT_128AW",
                    "baseline_FESC_128AW",
                    "baseline_RAVDESS_128AW",
                    "baseline_TIMIT_128AW",
                    "baseline_FINNISH_DIALECT_128AW",
                    ]

if conf.experiment is None:
    [ "\n"+str(i)+" "+exp   for i,exp in enumerate(list_of_experiments)]
    input_experiment=input("while experiments would want to do (enter numbers)? \n "+" ".join([ "\n"+str(i)+" "+exp   for i,exp in enumerate(list_of_experiments)])+"\n")
    if int(input_experiment) >= 0 and  int(input_experiment)<len(list_of_experiments):
        conf.experiment=list_of_experiments[int(input_experiment)]
    else:
        print("wrong experiment --program terminates")
        sys.exit(-1)


conf.run_name=conf.run_name_prefix+sys.argv[1]+conf.time_string
conf.description=sys.argv[2]

    

torch.manual_seed(conf.manual_seed)
#specification variables
audio_window=conf.audio_window
use_cuda=conf.use_cuda
timestep=conf.timestep
batch_size=conf.batch_size
logging_dir=conf.log_path+"logs/"+conf.experiment+"/"
epochs=conf.epochs
train_split=conf.train_split
run_name = conf.run_name

create_experiment_folder(logging_dir)
logger = setup_logs(logging_dir, run_name)
summary_writer_dir="runs/"
if "trail" in sys.argv[1]:
    summary_writer_dir="trail_runs/"
summary_writer_dir=conf.log_path+summary_writer_dir+conf.experiment+"/"
create_experiment_folder(summary_writer_dir)    
writer = SummaryWriter(summary_writer_dir+conf.run_name) 

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
model=None
optimizer=None
if conf.model is not None:
    if conf.train_ds_model or conf.validate_ds_model:
        model=conf.model(conf.ds_model_config,conf.ds_model_no_class).to(device)
    else:
        model=conf.model(project_no_projection=conf.timestep).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    if conf.load_model is not None:
         checkpoint = torch.load(conf.load_model, map_location=lambda storage, loc: storage)
         model.load_state_dict(checkpoint['state_dict'],strict=conf.is_model_load_strict)
         model=model.to(device)
         logger.info("parameters loaded for the upstream model "+str(model.__class__.__name__)+" from the file "+conf.load_model) 
    for name,param in model.named_parameters():
        for i in conf.named_parameters_to_ignore:
            if i in name  : 
                param.requires_grad=False
                print(name)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('===> up stream Model summary below###\n {}\n'.format(str(model)))
    logger.info('===> up stream total trainable parameter: {}\n'.format(model_params))
else:
    logger.info("WARNING::: up stream model is NONE")


args={"log_interval":conf.log_interval,"logging_dir":logging_dir,"epochs":epochs,"use_gpu":use_cuda,"device":device,"lr":conf.lr}

if conf.train_ds_model:
    args["loss_fn"]=conf.ds_loss_fn


global_timer = timer()

def write_model_to_tf(model,input_channel,input_length,device):
    temp_model = copy.deepcopy(model).to(device=device)
    temp_model.eval()
    data=torch.rand(1, input_channel,input_length).to(device=device)
    torch.onnx.export(model, data, conf.log_path+"model_vis/"+conf.run_name, verbose=True, input_names=["log_mel_speech"], output_names=["loss,output"])
    del temp_model

def process_training(model,epochs,args,train_loader,validation_loader,optimizer,batch_size,patience_thresold):
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    patience_counter=0 
    
    #if this training is the part of some previous training.
    for epoch in range(1, epochs + 1):
        if conf.training_history is not None and epoch in conf.training_history.keys():
            writer.add_scalar("loss/train",conf.training_history[epoch] , epoch)
            continue
        
        writer.flush()
        epoch_timer = timer()
        train_acc=train(args,model, train_loader, optimizer, epoch, batch_size,conf.train_ds_model)
        val_loss,val_acc = validation(args, model,validation_loader, batch_size,conf.validate_ds_model)
        writer.add_scalar("loss/train",val_loss , epoch)
        
        if train_acc is not None:
            writer.add_scalar("acc/train",train_acc , epoch)
        if val_acc is not None:
            writer.add_scalar("acc/val",val_acc , epoch)
        writer.flush()

        # Save
        if val_loss < best_loss: 
            best_loss = val_loss
            if conf.save_model and model is not None:
                snapshot(conf.log_path+'models/', run_name, { 
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'validation_loss': val_loss,
                    'optimizer': optimizer.state_dict(),
                })
                writer.add_text('model_stored_on', str(os.path.join(conf.log_path+'models/',run_name + '-model_best.pth')))
        
            best_epoch = epoch + 1
            patience_counter=0
        elif patience_counter>=patience_thresold:
            logger.info("maximum patience reached with no improvement breaking the training at epoch "+str(epoch+1) )
            break
        else:
            patience_counter=patience_counter+1 
        
    end_epoch_timer = timer()
    logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args["epochs"], end_epoch_timer - epoch_timer))


#write_model_to_tf(model,conf.channel,conf.audio_window,device)

if conf.train:
    train_loader,validation_loader=get_dataloaders(conf,tensor_writer=writer)
    print("validation size "+str(len(validation_loader.dataset)))
    print("train size "+str(len(train_loader.dataset)))
    writer.add_text('validation_data_size',str(len(validation_loader.dataset)))
    writer.add_text('train_data_size',str(len(train_loader.dataset)))
    #process_training(model,epochs,args,train_loader,validation_loader,optimizer,batch_size,conf.patience_thresold)
if conf.test:
    test_loader=get_dataloaders(conf)

    target,predicted=test_down_stream(args,model, test_loader, batch_size)
    
    clasifi_report=calculate_stats(target,predicted,target_names=conf.target_names)
    writer.add_text('classification_report', str(tabulate(clasifi_report, headers='keys', tablefmt='html')))

end_global_timer = timer()
logger.info("################## Success #########################")
logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))