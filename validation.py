import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch import nn 
from sklearn.metrics import accuracy_score

## Get the same logger from main"
logger = logging.getLogger("cdc")


def validation(args, model, data_loader, batch_size,is_validate_ds_model):    
    device=args["device"]
    logger.info("Starting Validation")
 
    model.eval()
    total_loss = 0
    target_output=[]
    y_true=[]
    acc=None
    with torch.no_grad():
        for idx,(data,y) in enumerate(data_loader):
                
            data = data.float().to(device) # add channel dimension 
            loss,output= model(data)
            if is_validate_ds_model:
                y=y.to(dtype=torch.long,device=device)
                loss=args["loss_fn"](output,y)
                predicted_y=torch.argmax(output,1)
                target_output.extend(predicted_y.cpu().detach().numpy().tolist())
                y_true.extend(y.cpu().detach().numpy().tolist())
            total_loss += len(data) * loss
             

    total_loss /= len(data_loader.dataset) # average loss

    
    if len(target_output)>0:
        acc=accuracy_score(y_true, target_output)
     
    logger.info("===> Validation set: Average loss: "+str(total_loss)+" ::: valiadation accuracy ==> "+str(acc))
    return total_loss,acc


def test_down_stream(args, model, data_loader, batch_size):
    device=args["device"]
    logger.info("Starting testing")
    for m in model.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm1d:
                #print("child updated")
                child.track_running_stats = False
                child.running_mean = None
                child.running_var = None
    model.eval()
    total_loss = 0
    total_acc  = 0 
    predicted_output=[]
    target_output=[]    
    #print("model.eval done ")
    with torch.no_grad():
    #with True:
    #print("torch.no_grad done ")
        for idx,(data,y) in enumerate(data_loader):
            #print(y)
            data = data.float().to(device) # add channel dimension
            target_output.extend(y.cpu().detach().numpy().tolist())
            loss,predicted_y=model(data)
            predicted_y=torch.argmax(predicted_y,1)
            predicted_output.extend(predicted_y.cpu().detach().numpy().tolist())
    return target_output,predicted_output