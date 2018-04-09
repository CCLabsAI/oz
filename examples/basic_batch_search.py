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

h = oz.make_leduk_history()
enc = oz.LedukEncoder()

bs = oz.BatchSearch(h, enc, 12)
batch = bs.generate_batch()
bigX = Variable(batch)

model = Net(enc.encoding_size())

logits = model.forward(bigX)
probs = logits.exp()
print(probs)
