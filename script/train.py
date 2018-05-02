import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import oz

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

rng = oz.Random()

history = oz.make_leduk_history()
encoder = oz.make_leduk_encoder()
target  = oz.make_leduk_target()

# n_cards = 6
# history = oz.make_goofspiel2_history(n_cards)
# encoder = oz.make_goofspiel2_encoder(n_cards)
# target  = oz.make_goofspiel2_target()

hidden_size = 25
search_batch_size = 80
eps = 0.1
delta = 0.9
gamma = 0.01

model = Net(input_size=encoder.encoding_size(),
            hidden_size=hidden_size,
            output_size=encoder.max_actions())

bs = oz.BatchSearch(batch_size=search_batch_size,
                    history=history,
                    encoder=encoder,
                    target=target,
                    eps=eps, delta=delta, gamma=gamma)

class Trainer:
    def train(self):
        pass

    def simulate(self):
        pass

def run_trainer_local(trainer):
    pass

def run_trainer_distributed(trainer):
    pass
