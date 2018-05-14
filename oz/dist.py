import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process

import numpy as np
import numpy.random

import oz.reservoir

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
        data = torch.zeros(batch_size, input_size)
        targets = torch.zeros(batch_size, output_size)

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


def run_trainer_distributed(trainer, args, size, start_iteration=0, iter_callback=None):
    reservoir_beta_ratio = args.reservoir_beta_ratio
    reservoir_size = args.reservoir_size

    encoding_size = trainer.encoder.encoding_size()
    max_actions = trainer.encoder.max_actions()
    sample_size = [reservoir_size, encoding_size + max_actions]
    reservoir = oz.reservoir.ExponentialReservoir(
                    sample_size=sample_size,
                    beta_ratio=reservoir_beta_ratio)

    def run(rank, size):
        iteration_n = start_iteration

        train_batch_size = args.train_batch_size
        train_game_ply = args.train_game_ply
        train_steps = args.train_steps
        train_iter = args.train_iter
        print("[{}/{}]: alive!".format(rank, size), flush=True)

        if rank == 0:
            while iteration_n < train_iter:
                broadcast_net(trainer.model)

                print("[{}/{}]: starting iter: {}".format(rank, size, iteration_n), flush=True)
                data = torch.zeros(train_batch_size, encoding_size)
                targets = torch.zeros(train_batch_size, max_actions)
                for j in range(train_game_ply):
                    infoset_encoding, action_probs = trainer.simulate()
                    data[j] = infoset_encoding
                    targets[j] = action_probs
                    if args.progress:
                        print(".", end="", flush=True)
                all_data, all_targets = gather_experience_rank0(size, data, targets)
                if args.progress:
                    print(flush=True)

                all_experience = torch.cat((all_data, all_targets), dim=1)
                for j in range(all_experience.size(0)):
                    reservoir.add(all_experience[j])

                losses  = torch.zeros(train_steps)
                sample  = reservoir.sample()
                data    = sample[:,:encoding_size]
                targets = sample[:,encoding_size:]

                for k in range(train_steps):
                    idx = np.random.choice(sample.size(0), args.train_batch_size)
                    idx = torch.from_numpy(idx)
                    x = data[idx]
                    y = targets[idx]
                    loss = trainer.train(x, y)
                    losses[k] = loss

                iteration_n += 1

                if iter_callback:
                    iter_callback(
                        iteration_n=iteration_n,
                        args=args,
                        interrupted=False,
                        trainer=trainer,
                        losses=losses)

        else:
            while iteration_n < train_iter:
                broadcast_net(trainer.model)

                print("[{}/{}]: starting iter: {}".format(rank, size, iteration_n), flush=True)
                data = torch.zeros(train_batch_size, encoding_size)
                targets = torch.zeros(train_batch_size, max_actions)
                for j in range(train_game_ply):
                    infoset_encoding, action_probs = trainer.simulate()
                    data[j] = infoset_encoding
                    targets[j] = action_probs
                    if args.progress:
                        print(".", end="", flush=True)
                gather_experience(data, targets)
                iteration_n += 1

    start_proceses(run, size)
