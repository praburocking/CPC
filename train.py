
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
        print(y.shape)
        y=y.to(dtype=torch.long) 
        data = data.float().to(device) # add channel dimension
        optimizer.zero_grad()
#        hidden = us_model.init_hidden(len(data), use_gpu=args["use_gpu"])
        #up stream task gets the representation
        print("upstream stream task started")
        loss,embeddings = us_model(data)
        #down stream task gets the representation and processes the classification
        print("down stream task started")
        predicted_y=ds_model(embeddings)
        print(type(y))
        #print(y)
        print(predicted_y.shape)
        print(y.shape)
        predicted_y=predicted_y.transpose(1,2)
        length=predicted_y.shape[-1]
        batch_size=predicted_y.shape[0]
        updated_y=torch.empty(batch_size,length,dtype=torch.long)
        for i in range(batch_size):
            updated_y[i,:]=updated_y[i,:].fill_(y[i])
        loss=args["down_stream_loss_fn"](predicted_y,updated_y)
        #print(y)
        #print(predicted_y)

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