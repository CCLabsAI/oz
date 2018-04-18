from copy import copy
import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import oz

class NetSigma(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NetSigma, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class NetRegret(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NetRegret, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x

root = oz.make_leduk_history()
h = copy(root)
enc = oz.LedukEncoder()
rng = oz.Random(1)

search_size = 50
bs = oz.BatchSearch(root, enc, search_size)
ex = 100
batch = None

encoding_size = enc.encoding_size()
max_actions = enc.max_actions()

torch.manual_seed(7)
net_sigma = NetSigma(input_size=encoding_size, hidden_size=18, output_size=max_actions)
net_regret = NetRegret(input_size=encoding_size, hidden_size=64, output_size=max_actions)

optimizer_sigma = optim.Adam(net_sigma.parameters(), lr=1e-3)
optimizer_regret = optim.Adam(net_regret.parameters(), lr=1e-2)

criterion_sigma = nn.KLDivLoss(size_average=True)
criterion_regret = nn.SmoothL1Loss(size_average=True)

def pr_nn(infoset, action):
    d = torch.zeros(encoding_size)
    enc.encode(infoset, d)
    sigma_logits = net_sigma.forward(Variable(d.unsqueeze(0)))
    sigma_pr = sigma_logits.exp()
    m = enc.decode(infoset, sigma_pr.data[0])
    return m[action]

sigma_nn = oz.make_py_sigma(pr_nn)
ex = oz.exploitability(h, sigma_nn)
print(ex)

# while ex > .1:
for i in range(100):
    n_iter = 1000
    for j in range(n_iter):
        batch = bs.generate_batch()
        if len(batch) == 0:
            bs.step(torch.Tensor(), torch.Tensor(), rng)
        else:
            batch_var = Variable(batch, volatile=True)
            sigma_logits = net_sigma.forward(batch_var)
            regret_est = net_regret.forward(batch_var)
            sigma_probs = sigma_logits.exp()
            bs.step(sigma_probs.data, 0*regret_est.data, rng)

    tree = bs.tree
    nodes = tree.nodes
    sigma = tree.sigma_average()
    bigX_list = []
    sigma_target_list = []
    regret_target_list = []
    for infoset, node in nodes.items():
        # if node.regret_n < 100:
        #     continue

        d = torch.zeros(encoding_size)
        enc.encode(infoset, d)
        bigX_list.append(d)

        pr = torch.Tensor([sigma.pr(infoset, a) for a in infoset.actions])
        if len(pr) == 2: # FIXME!!
            pr = torch.cat([torch.zeros(1), pr])
        sigma_target_list.append(pr)

        n = node.regret_n
        # r = torch.Tensor([r / n if n > 0 else 0 for a, r in node.regrets.items()])
        r = torch.Tensor([r / n_iter for a, r in node.regrets.items()])
        if len(r) == 2: # FIXME!!
            r = torch.cat([torch.zeros(1), r])
        regret_target_list.append(r)

    bigX = torch.stack(bigX_list)
    sigma_target = torch.stack(sigma_target_list)
    regret_target = torch.stack(regret_target_list)

    bigX_var = Variable(bigX)
    sigma_target_var = Variable(sigma_target)
    regret_target_var = Variable(regret_target)

    for j in range(500):
        optimizer_sigma.zero_grad()
        sigma_logits = net_sigma(bigX_var)
        loss_sigma = criterion_sigma(sigma_logits, sigma_target_var)
        loss_sigma.backward()
        optimizer_sigma.step()

        optimizer_regret.zero_grad()
        regret_est = net_regret(bigX_var)
        loss_regret = criterion_regret(regret_est, regret_target_var)
        loss_regret.backward()
        optimizer_regret.step()

    # print(regret)
    # print(regret_target_var)

    # TODO implement a method to "drain" the batch search
    loss_sigma0 = float(loss_sigma.data)
    loss_regret0 = float(loss_regret.data)
    ex = oz.exploitability(h, sigma)
    ex_nn = oz.exploitability(h, sigma_nn)
    print("%.3f, %.3f, %.5f, %.5f" % (ex, ex_nn, loss_sigma0, loss_regret0))

    if i > 0 and (i % 50) == 0:
        bs = oz.BatchSearch(root, enc, search_size)
        print("reset!")

tree = bs.tree
nodes = tree.nodes
sigma = tree.sigma_average()

infoset_regrets = [
    (str(infoset), node.regret_n,
        sorted([(a.index, v / node.regret_n if node.regret_n > 0 else 0, sigma.pr(infoset, a)) for a, v in node.regrets.items()]),
        sorted([(a.index, v) for a, v in node.regrets.items()]),
        sorted([(a.index, s) for a, s in node.average_strategy.items()]))
    for infoset, node in nodes.items()
]

for infoset_str, n, action_targets, action_regrets, action_avg in sorted(infoset_regrets):
    print(infoset_str, n, action_targets)
    # print(infoset_str, n, action_targets, action_regrets, action_avg)
