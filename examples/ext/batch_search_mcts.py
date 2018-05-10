import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import oz

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 25)
        self.fc2 = nn.Linear(25, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

rng = oz.Random(1)

history = oz.make_leduk_history()
enc = oz.make_leduk_encoder()

params = oz.MCTSParams()

params.c = 18
params.eta = 0.9
params.gamma = 0.1
params.d = 0.002
params.smooth = True

bs = oz.MCTSBatchSearch(20, history, enc, params)
model = Net(enc.encoding_size())

for i in range(1000000):
    batch = bs.generate_batch()

    if len(batch) == 0:
        bs.step(rng)
    else:
        batch.requires_grad = False
        logits = model.forward(batch)
        probs = logits.exp()
        bs.step(probs, rng)

    if i % 10000 == 0:
        tree = bs.tree
        sigma = tree.sigma_average()
        print("%.5f" % oz.exploitability(history, sigma))
