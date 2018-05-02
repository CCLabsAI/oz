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

# n_cards = 13
# history = oz.make_goofspiel2_history(n_cards)
# encoder = oz.make_goofspiel2_encoder(n_cards)
# target  = oz.make_goofspiel2_target()

history = oz.make_leduk_history()
encoder = oz.make_leduk_encoder()
target  = oz.make_leduk_target()

model = Net(input_size=encoder.encoding_size(),
            hidden_size=25,
            output_size=encoder.max_actions())

bs = oz.BatchSearch(80, history, encoder, target,
                    eps=0.2, delta=.9, gamma=0.01)

while not history.is_terminal():
    if history.player == oz.Chance:
        ap = history.sample_chance(rng)
        history.act(ap.a)

    else:
        infoset = history.infoset()
        bs.target(infoset)

        for i in range(10000):
            batch = bs.generate_batch()
            batch.requires_grad = False

            if len(batch) == 0:
                bs.step(rng)
            else:
                logits = model.forward(batch)
                probs = logits.exp()
                bs.step(probs, rng)

        tree = bs.tree
        sigma = tree.sigma_average()
        node = bs.tree.lookup(infoset)

        print([(a.index, sigma.pr(infoset, a)) for a in infoset.actions])

        ap = sigma.sample_pr(infoset, rng)
        print("action: ", ap.a.index)
        print("node visits: ", node.regret_n, history.player)
        print("average targeting ratio: ", bs.avg_targeting_ratio)
        history.act(ap.a)

print(probs)
print(history)
