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

h = oz.make_leduc_history()
enc = oz.make_leduc_encoder()

bs = oz.BatchSearch(12, h, enc)
batch = bs.generate_batch()

while len(batch) == 0:
    bs.step(torch.zeros(0), rng)
    batch = bs.generate_batch()

batch.requires_grad = False

model = Net(enc.encoding_size())

logits = model.forward(batch)
probs = logits.exp()
print(probs)
