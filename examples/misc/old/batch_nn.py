from copy import copy

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import oz


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


root = oz.make_leduc_history()
h = copy(root)
enc = oz.LeducEncoder()
rng = oz.Random(1)

search_size = 50
bs = oz.BatchSearch(root, enc, search_size)
ex = 100
batch = None

encoding_size = enc.encoding_size()
max_actions = enc.max_actions()

torch.manual_seed(7)
net = Net(input_size=encoding_size, hidden_size=18, output_size=max_actions)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
criterion = nn.KLDivLoss(size_average=True)

while ex > .25:
    for i in range(1000):
        batch = bs.generate_batch()
        if len(batch) == 0:
            bs.step(torch.Tensor(), rng)
        else:
            logits = net.forward(Variable(batch, volatile=True))
            probs = logits.exp()
            bs.step(probs.data, rng)

    tree = bs.tree
    sigma = tree.sigma_average()
    bigX_list = []
    bigY_list = []
    for infoset, node in tree.nodes.items():
        d = torch.zeros([encoding_size])
        enc.encode(infoset, d)
        bigX_list.append(d)
        pr = torch.Tensor([sigma.pr(infoset, a) for a in infoset.actions])
        if len(pr) == 2: # FIXME!!
            pr = torch.cat([torch.zeros(1), pr])
        bigY_list.append(pr)

    bigX = torch.stack(bigX_list)
    bigY = torch.stack(bigY_list)

    bigX_var = Variable(bigX)
    bigY_var = Variable(bigY)

    for i in range(100):
        optimizer.zero_grad()
        output = net(bigX_var)
        loss = criterion(output, bigY_var)
        loss.backward()
        optimizer.step()

    # TODO implement a method to "drain" the batch search
    loss0 = float(loss.data)
    ex = oz.exploitability(h, sigma)
    print("%.3f, %.3f" % (ex, loss0))
