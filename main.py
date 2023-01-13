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
from dataset import RawDataset
from model import CDCK2
from scheduledoptimizer import ScheduledOptim
from logger import setup_logs
from train import train,snapshot
from validation import validation

dev_outputfile_name='dev-Librispeech.h5'
dev_outputlist_name='dev-Librispeech.pkl'
test_outputfile_name='test-Librispeech.h5'
test_outputlist_name='test-Librispeech.pkl'

#specification variables
audio_window=2048
use_cuda=True
timestep=12
batch_size=64
audio_window=20480
warmup_steps=20
logging_dir='./logs'
epochs=100
train_split=0.9
run_name = "cdc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
logger = setup_logs(logging_dir, run_name) 

training_set   = RawDataset(dev_outputfile_name, dev_outputlist_name, audio_window)
device = torch.device("cuda" if use_cuda else "cpu")
model = CDCK2(timestep, batch_size, audio_window).to(device)

optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        warmup_steps)

model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


logger.info('===> loading train, validation and eval dataset')
training_set   = RawDataset(dev_outputfile_name, dev_outputlist_name, audio_window)
no_training_data=int(len(training_set)*train_split)
no_val_data=int(len(training_set)-no_training_data)

test_set   = RawDataset(test_outputfile_name, test_outputlist_name, audio_window)

training_set, validation_set = torch.utils.data.random_split(training_set, [no_training_data, no_val_data])

train_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
validation_loader = data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

logger.info('### Model summary below###\n {}\n'.format(str(model)))
logger.info('===> Model total parameter: {}\n'.format(model_params))

args={"log_interval":100,"logging_dir":logging_dir,"epochs":epochs}

## Start training
global_timer = timer()
best_acc = 0
best_loss = np.inf
best_epoch = -1 
for epoch in range(1, epochs + 1):
    epoch_timer = timer()

    train(args, model, device, train_loader, optimizer, epoch, batch_size)
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

## end 
end_global_timer = timer()
logger.info("################## Success #########################")
logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))