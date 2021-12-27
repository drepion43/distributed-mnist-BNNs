import os
import time
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from utils import *
import torch.optim as optim
from torch.autograd import Variable

from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from models.binarized_modules import  Binarize,HingeLoss
import pandas as pd
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda=torch.cuda.is_available()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '192.168.0.2'
    os.environ['MASTER_PORT'] = '23456'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 1024*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(1024*self.infl_ratio)
        self.fc2 = BinarizeLinear(1024*self.infl_ratio, 512*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(512*self.infl_ratio)
        self.fc3 = BinarizeLinear(512*self.infl_ratio, 256*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(256*self.infl_ratio)
        self.fc4 = nn.Linear(256*self.infl_ratio, 10)
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
    T=[]
    E=[]
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=rank)
    torch.cuda.manual_seed(args.seed)
    model = Net()
    torch.cuda.set_device(0)
    model.cuda()
    batch_size = 64
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
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
    starts = datetime.now()
    total_step = len(train_loader)
    for epoch in range(1,args.epochs+1):
        T.append(["epoch",epoch])
        start = datetime.now()
        train_time = AverageMeter()
        end = time.time()
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            if epoch%40==0:
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

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
            if batch_idx % args.log_interval == 0:
                if( batch_idx * len(data)) !=0:
                    T.append([batch_idx * len(data),train_time.val])
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Time: {:.3f}({:.3f})'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),train_time.val,train_time.avg))
        print("Training ",epoch,' : ' + str(datetime.now() - start))
        E.append([datetime.now() - start])
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - starts))

    dataframe=pd.DataFrame(T)
    dataframe.to_csv("MNIST_BATCH_TIME(PersonalCom).csv")
    dataframes=pd.DataFrame(E)
    dataframes.to_csv("MNIST_EPOCH_TIME(PersonalCom).csv")


if __name__ == '__main__':
    main()
