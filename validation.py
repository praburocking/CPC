import numpy as np
import logging
import torch
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("cdc")


def validation(args, model,data_loader, batch_size):
    device=args["device"]
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc=0

    with torch.no_grad():
        for data in data_loader:
            data = data.float().to(device) # add channel dimension
          #  hidden = model.init_hidden(len(data), use_gpu=args["use_gpu"])
            loss,_ = model(data)
            total_loss += len(data) * loss 
          

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\n'.format(
                total_loss))

    return total_acc, total_loss



def validation_down_stream(args, us_model,ds_model, data_loader, batch_size):
    device=args["device"]
    logger.info("Starting Validation")
    us_model.eval()
    ds_model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for idx,(data,y) in enumerate(data_loader):
            data = data.float().to(device) # add channel dimension
            y=y.to(dtype=torch.long) 
            
            loss,embeddings= us_model(data)
            predicted_y=ds_model(embeddings)
            loss=args["down_stream_loss_fn"](predicted_y,y)
            total_loss += len(data) * loss 

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss