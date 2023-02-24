
import torch
import logging
import os
import torch.nn.functional as F
logger = logging.getLogger("cdc")

def train(args, model, train_loader, optimizer, epoch, batch_size):
    device=args["device"]
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.float().to(device) # add channel dimension
        optimizer.zero_grad()
#        hidden = model.init_hidden(len(data), use_gpu=args["use_gpu"])
        loss,_ = model(data)
        acc=0
        loss.backward()
        optimizer.step()
       # lr = optimizer.update_learning_rate()
        lr=args["lr"]
        if batch_idx % args["log_interval"] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))

def train_down_stream(args,us_model, ds_model, train_loader, optimizer, epoch, batch_size):
    device=args["device"]
    ds_model.train()
    us_model.eval()
    for batch_idx, (data,y) in enumerate(train_loader):
        y=torch.Tensor(y)
        y=y.to(dtype=torch.long,device=device) 
        data = data.float().to(device) # add channel dimension
        optimizer.zero_grad()
#        hidden = us_model.init_hidden(len(data), use_gpu=args["use_gpu"])
        #up stream task gets the representation
       # print("upstream stream task started")
        loss,embeddings = us_model(data)
        #down stream task gets the representation and processes the classification
        #print("down stream task started")
        predicted_y=ds_model(embeddings)
        loss=args["down_stream_loss_fn"](predicted_y,y)

        loss.backward()
        optimizer.step()

        acc=0
        if batch_idx % args["log_interval"] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), args["lr"], acc, loss.item()))

def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))