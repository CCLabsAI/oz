#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

n_iter = 50
batch_size = 50
input_size = 100
output_size = 3

def simulate(net, data, targets):
    data *= 0.9
    targets *= 0.9

def train(net, all_train_data, all_train_targets, optimizer, criterion):
    train_data_var = Variable(all_train_data)
    train_targets_var = Variable(all_train_targets)
    
    optimizer.zero_grad()
    output = net.forward(train_data_var);
    loss = criterion(output, train_targets_var)
    loss.backward()
    optimizer.step()

    print(loss.data[0])

def gather_experience_rank0(size, data, targets):
    data_gather_list = [torch.zeros(batch_size, input_size)
                           for i in range(size)]

    target_gather_list = [torch.zeros(batch_size, output_size)
                             for i in range(size)]

    dist.gather(data, dst=0, gather_list=data_gather_list)
    dist.gather(targets, dst=0, gather_list=target_gather_list)
    
    all_train_data = torch.cat(data_gather_list)
    all_train_targets = torch.cat(target_gather_list)
    
    return all_train_data, all_train_targets

def gather_experience(data, targets):
    dist.gather(data, dst=0)
    dist.gather(targets, dst=0)

def broadcast_net(net):
    for param in net.parameters():
        dist.broadcast(param.data, src=0)    

def run(rank, size):
    net = Net(input_size=input_size, hidden_size=18, output_size=output_size)
    data = torch.randn(batch_size, input_size)
    targets = torch.randn(batch_size, output_size)

    if rank == 0:
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        criterion = nn.MSELoss(size_average=True)

        for i in range(n_iter):
            simulate(net, data, targets)
            all_data, all_targets = gather_experience_rank0(size, data, targets)
            train(net, all_data, all_targets, optimizer, criterion)
            broadcast_net(net)
    
    else:
        for i in range(n_iter):
            simulate(net, data, targets)
            gather_experience(data, targets)
            broadcast_net(net)

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 9
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
