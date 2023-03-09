
import torch
import logging
import os
import torch.nn.functional as F
logger = logging.getLogger("cdc")


def train(args,us_model, ds_model, train_loader, optimizer, epoch, batch_size,is_train_us_model,is_train_ds_model):
    device=args["device"]
    assert is_train_us_model or is_train_ds_model, "No training flags are found, check the conf.py file"
    if is_train_us_model:
        assert  us_model is not None, "if we need to train us_model, us_model has to be set in conf.py file"
    if is_train_ds_model:
        assert ds_model is not None , "if we need to train ds_model, us_model and ds_model has to be set in conf.py file"
  

    if us_model  is not None: 
        if is_train_us_model: 
            us_model.train()
        else:
            us_model.eval()
    if ds_model  is not None: 
        if is_train_ds_model: 
            ds_model.train()
        else:
            ds_model.eval()
    for batch_idx, data in enumerate(train_loader):
        y=None
        optimizer.zero_grad()

        # split data and label if we need to train down_stream 
        if is_train_ds_model:
            data,y=data
            y=torch.Tensor(y).to(dtype=torch.long,device=device)
        
        data = data.float().to(device) # add channel dimension
        loss,embeddings = us_model(data)

        # if is train down stream then try to pass the embeddings to downstream model too.
        if is_train_ds_model:
            predicted_y=ds_model(embeddings)
            loss=args["down_stream_loss_fn"](predicted_y,y)

        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), args["lr"], loss.item()))

def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))