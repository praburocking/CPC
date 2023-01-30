import numpy as np
import logging
import torch
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("cdc")


def validation(args, model, device, data_loader, batch_size):
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for data in data_loader:
            data = data.float().unsqueeze(1).to(device) # add channel dimension
            hidden = model.init_hidden(len(data), use_gpu=args["use_gpu"])
            acc, loss, hidden = model(data, hidden)
            total_loss += len(data) * loss 
            total_acc  += len(data) * acc

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss



def validation_down_stream(args, model, device, data_loader, batch_size):
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for data in data_loader:
            data = data.float().unsqueeze(1).to(device) # add channel dimension
            hidden = model.init_hidden(len(data), use_gpu=args["use_gpu"])
            acc, loss, hidden = model(data, hidden)
            total_loss += len(data) * loss 
            total_acc  += len(data) * acc

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss