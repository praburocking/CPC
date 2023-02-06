
import torch
import logging
import os
import torch.nn.functional as F
logger = logging.getLogger("cdc")

def train(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.float().to(device) # add channel dimension
        optimizer.zero_grad()
#        hidden = model.init_hidden(len(data), use_gpu=args["use_gpu"])
        loss = model(data)
        acc=0
        loss.backward()
        optimizer.step()
       # lr = optimizer.update_learning_rate()
        lr=args["lr"]
        if batch_idx % args["log_interval"] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))

def train_down_stream(args,us_model, ds_model, device, train_loader, optimizer, epoch, batch_size):
    ds_model.train()
    us_model.eval()
    for batch_idx, (data,y) in enumerate(train_loader):
        data = data.float().to(device) # add channel dimension
        optimizer.zero_grad()
        hidden = us_model.init_hidden(len(data), use_gpu=args["use_gpu"])
        #up stream task gets the representation
        acc, loss, hidden = us_model(data, hidden)
        #down stream task gets the representation and processes the classification
        loss=ds_model(data)
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % args["log_interval"] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))

def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))