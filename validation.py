import numpy as np
import logging
import torch
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("cdc")


def validation(args, model, data_loader, batch_size,is_validate_ds_model):    
    device=args["device"]
    logger.info("Starting Validation")
 
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for idx,(data,y) in enumerate(data_loader):
                
            data = data.float().to(device) # add channel dimension 
            loss,output= model(data)
            if is_validate_ds_model:
                y=y.to(dtype=torch.long,device=device)
                loss=args["loss_fn"](output,y)
            total_loss += len(data) * loss 

    total_loss /= len(data_loader.dataset) # average loss

    logger.info('===> Validation set: Average loss: {:.4f}\n'.format(total_loss))

    return total_loss


def test_down_stream(args, model, data_loader, batch_size):
    device=args["device"]
    logger.info("Starting testing")
    model.eval()
    total_loss = 0
    total_acc  = 0 
    predicted_output=[]
    target_output=[]    
    with torch.no_grad():
        for idx,(data,y) in enumerate(data_loader):
            data = data.float().to(device) # add channel dimension
            target_output.extend(y.cpu().detach().numpy().tolist())
            loss,predicted_y=model(data)
            predicted_y=torch.argmax(predicted_y,1)
            predicted_output.extend(predicted_y.cpu().detach().numpy().tolist())
    return predicted_output,target_output