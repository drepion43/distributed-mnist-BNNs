import os
import time
from datetime import datetime
import argparse

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn

# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from models.binarized_modules import  Binarize,HingeLoss
from utils import *
import numpy as np
import pandas as pd


# def main():

#     os.environ['MASTER_ADDR'] = '192.168.0.19'
#     os.environ['MASTER_PORT'] = '23456'
#     mp.spawn(train, nprocs=args.gpus, args=(args,))


class Net(nn.Module):
    def __init__(self,dev0,dev1):
        super(Net, self).__init__()
        self.dev0=dev0
        self.dev1=dev1
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 1024*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(1024*self.infl_ratio).to(dev0)
        self.fc2 = BinarizeLinear(1024*self.infl_ratio, 512*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(512*self.infl_ratio).to(dev1)
        self.fc3 = BinarizeLinear(512*self.infl_ratio, 256*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(256*self.infl_ratio).to(dev0)
        self.fc4 = nn.Linear(256*self.infl_ratio, 10).to(dev1)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    model = Net()
    torch.cuda.set_device(0)
    model.cuda()
    batch_size = 64
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    T=[]
    E=[]
    starts = datetime.now()
    total_step = len(train_loader)
    for epoch in range(1, args.epochs+1):
        T.append(["epoch",epoch])
        start = datetime.now()
        train_time = AverageMeter()
        end = time.time()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            if epoch%40==0:
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()

            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))
            train_time.update(time.time() - end)
            end=time.time()
            
            if batch_idx % args.log_interval == 0 and gpu == 0:
                if( batch_idx * len(data)) !=0:
                    T.append([batch_idx * len(data),train_time.val])
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Time={:.3f}({:.3f})'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(),train_time.val,train_time.avg))
        E.append([datetime.now() - start])
            # if (i + 1) % 100 == 0 :
            #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
            #                                                              loss.item()))
    
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - starts))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.2'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = Net(dev0,dev1).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()

def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = (rank * 2) % world_size
    dev1 = (rank * 2 + 1) % world_size
    mp_model = Net(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = Net().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    args = parser.parse_args()
    world_size = args.gpus * args.nodes
    rank=args.nr * args.gpus
    run_demo(demo_basic, world_size)
    run_demo(demo_checkpoint, world_size)
    run_demo(demo_model_parallel, world_size)
