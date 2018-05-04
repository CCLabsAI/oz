import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process

RunSpec = namedtuple('RunSpec', ['n_iter', 'batch_size', 'input_size', 'output_size'])


def gather_experience_rank0(size, data, targets):
    data_gather_list = [torch.zeros_like(data)
                           for i in range(size)]

    target_gather_list = [torch.zeros_like(targets)
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


def make_runner(runspec, net, train, simulate):
    n_iter = runspec.n_iter
    batch_size = runspec.batch_size
    input_size = runspec.input_size
    output_size = runspec.output_size

    def run(rank, size):
        data = torch.randn(batch_size, input_size)
        targets = torch.randn(batch_size, output_size)

        targets.abs_()
        targets /= targets.sum(dim=1, keepdim=True)

        if rank == 0:
            optimizer = optim.Adam(net.parameters(), lr=1e-3)
            criterion = nn.KLDivLoss(size_average=True)

            broadcast_net(net)
            for i in range(n_iter):
                simulate(net, data, targets)
                all_data, all_targets = gather_experience_rank0(size, data, targets)
                train(net, all_data, all_targets, optimizer, criterion)
                broadcast_net(net)

        else:
            broadcast_net(net)
            for i in range(n_iter):
                simulate(net, data, targets)
                gather_experience(data, targets)
                broadcast_net(net)

    return run


def init_processes(rank, size, fn, backend='tcp'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def start_proceses(run, size):
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

