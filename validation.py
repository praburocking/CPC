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
           # hidden = model.init_hidden(len(data), use_gpu=args["use_gpu"])
            loss,embeddings= us_model(data)
            predicted_y=ds_model(embeddings)
            predicted_y=predicted_y.transpose(1,2)
            length=predicted_y.shape[-1]
            batch_size=predicted_y.shape[0]
            updated_y=torch.empty(batch_size,length,dtype=torch.long)
            for i in range(batch_size):
                updated_y[i,:]=updated_y[i,:].fill_(y[i])
            loss=args["down_stream_loss_fn"](predicted_y,updated_y)
            total_loss += len(data) * loss 

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss