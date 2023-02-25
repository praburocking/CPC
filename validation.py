import numpy as np
import logging
import torch
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("cdc")


def validation_old(args, model,data_loader, batch_size,is_validate_us_model,is_validate_ds_model):
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



def validation(args, us_model,ds_model, data_loader, batch_size,is_validate_us_model,is_validate_ds_model):

    assert is_validate_ds_model or is_validate_us_model, "no validation flag found"
    if is_validate_us_model:
        assert  us_model is not None, "if we want to validate us_model then us_model has to be set in conf.py"
    if is_validate_ds_model:
        assert ds_model is not None and us_model is not None, "if we want to validate us_model then us_model and ds_model has to be set in conf.py"
   
    device=args["device"]
    logger.info("Starting Validation")
    
    if us_model is not None:
        us_model.eval()
    if ds_model is not None:
        ds_model.eval()
    total_loss = 0

    with torch.no_grad():
        for idx,data in enumerate(data_loader):
            y=None
            if is_validate_ds_model:
                data,y=data
                y=y.to(dtype=torch.long,device=device)
            data = data.float().to(device) # add channel dimension 
            loss,embeddings= us_model(data)
            if is_validate_ds_model:
                predicted_y=ds_model(embeddings)
                loss=args["down_stream_loss_fn"](predicted_y,y)
            total_loss += len(data) * loss 

    total_loss /= len(data_loader.dataset) # average loss

    logger.info('===> Validation set: Average loss: {:.4f}\n'.format(total_loss))

    return total_loss


def test_down_stream(args, us_model,ds_model, data_loader, batch_size):
    device=args["device"]
    logger.info("Starting testing")
    us_model.eval()
    ds_model.eval()
    total_loss = 0
    total_acc  = 0 
    predicted_output=[]
    target_output=[]    
    with torch.no_grad():
        for idx,(data,y) in enumerate(data_loader):
            data = data.float().to(device) # add channel dimension
            target_output.extend(y.cpu().detach().numpy().tolist())
            loss,embeddings= us_model(data)
            predicted_y=ds_model(embeddings)
            predicted_y=torch.argmax(predicted_y,1)
            predicted_output.extend(predicted_y.cpu().detach().numpy().tolist())
    return predicted_output,target_output