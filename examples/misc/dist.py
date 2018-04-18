from copy import copy
import random

import oz
import oz.dist

import torch
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

search_size = 50
batch_size = 100

def make_simulate():
    root = oz.make_leduk_history()
    enc = oz.LedukEncoder()
    rng = oz.Random() # TODO seed based on rank

    bs = oz.BatchSearch(root, enc, search_size)
    empty_tensor = torch.Tensor()

    def simulate(net, data, targets):
        n_iter = 5000
        for j in range(n_iter):
            batch = bs.generate_batch()
            if len(batch) == 0:
                bs.step(empty_tensor, empty_tensor, rng)
            else:
                batch_var = Variable(batch, volatile=True)
                sigma_logits = net.forward(batch_var)
                regret_est = torch.zeros_like(sigma_logits)
                sigma_probs = sigma_logits.exp()
                bs.step(sigma_probs.data, regret_est.data, rng)

        tree = bs.tree
        sigma = tree.sigma_average()
        node_list = random.sample(tree.nodes.items(), batch_size)
        bigX_list = []
        sigma_target_list = []
        
        for infoset, node in node_list:
            d = torch.zeros(encoding_size)
            enc.encode(infoset, d)
            bigX_list.append(d)

            pr = torch.Tensor([sigma.pr(infoset, a) for a in infoset.actions])
            if len(pr) == 2: # FIXME!!
                pr = torch.cat([torch.zeros(1), pr])
            sigma_target_list.append(pr)

        torch.stack(bigX_list, out=data)
        torch.stack(sigma_target_list, out=targets)


    return simulate

def train(net, all_data, all_targets, optimizer, criterion):
    data_var = Variable(all_data)
    targets_var = Variable(all_targets)
    
    for i in range(50):
        optimizer.zero_grad()
        output = net(data_var);
        loss = criterion(output, targets_var)
        loss.backward()
        optimizer.step()

    def pr_nn(infoset, action):
        d = torch.zeros(encoding_size)
        enc.encode(infoset, d)
        sigma_logits = net.forward(Variable(d.unsqueeze(0)))
        sigma_pr = sigma_logits.exp()
        m = enc.decode(infoset, sigma_pr.data[0])
        return m[action]
    sigma_nn = oz.make_py_sigma(pr_nn)
    h = oz.make_leduk_history()

    loss_val = loss.data[0]
    ex_nn = oz.exploitability(h, sigma_nn)
    print("%.3f, %.5f" % (ex_nn, loss_val))

if __name__ == "__main__":
    enc = oz.LedukEncoder()

    encoding_size = enc.encoding_size()
    max_actions = enc.max_actions()

    runspec = oz.dist.RunSpec(n_iter=500, batch_size=batch_size, input_size=encoding_size, output_size=max_actions)
    net = Net(input_size=runspec.input_size, hidden_size=256, output_size=runspec.output_size)
    simulate = make_simulate()

    # simulate(net, None, None)
    run = oz.dist.make_runner(runspec, net, train, simulate)
    oz.dist.start_proceses(run, size=4)

