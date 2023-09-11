
import torch
import logging
import os
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
logger = logging.getLogger("cdc")


def train(args,model, train_loader, optimizer, epoch, batch_size,is_train_ds_model):
    device=args["device"]

    model.train()
    target_output=[]
    y_true=[]
    for batch_idx, (data,y) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.float().to(device) # add channel dimension
        loss,output = model(data)
       
        if is_train_ds_model and y is not None:
            y=torch.Tensor(y).to(dtype=torch.long,device=device)
            loss=args["loss_fn"](output,y)
            predicted_y=torch.argmax(output,1)
            target_output.extend(predicted_y.cpu().detach().numpy().tolist())
            y_true.extend(y.cpu().detach().numpy().tolist())
            

        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), args["lr"], loss.item()))
            
    if len(target_output)>0:    
        acc=accuracy_score(y_true, target_output)
        return acc
    return None
        
            
            
            

def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))