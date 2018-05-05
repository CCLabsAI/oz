import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data

import oz
import oz.reservoir

from copy import copy


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

# n_cards = 13
# history = oz.make_goofspiel2_history(n_cards)
# encoder = oz.make_goofspiel2_encoder(n_cards)
# target  = oz.make_goofspiel2_target()

hidden_size = 64
search_batch_size = 20
play_eps = 0.25
eps = 0.4
delta = 0.9
gamma = 0.01
learning_rate = 0.1
n_simulation_iter = 10000

beta_ratio = 2.0
reservoir_size = 2**17
n_game_steps = 256
train_batch_size = 256
n_train_iter = 256

model = Net(input_size=encoder.encoding_size(),
            hidden_size=hidden_size,
            output_size=encoder.max_actions())

def make_batch_search():
    return oz.BatchSearch(batch_size=search_batch_size,
                          history=history,
                          encoder=encoder,
                          target=target,
                          eps=eps, delta=delta, gamma=gamma)

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.KLDivLoss(size_average=False)


class Trainer:
    def __init__(self, history, make_batch_search, encoder,
                 model, optimizer, criterion,
                 n_simulation_iter, rng):
        self.root_history = copy(history)
        self.history = copy(history)
        self.make_batch_search = make_batch_search
        self.batch_search = make_batch_search()
        self.encoder = encoder
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_simulation_iter = n_simulation_iter
        self.rng = rng

    def train(self, data, targets):
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        batch_size = data.shape[0]

        optimizer.zero_grad()
        output = model(data);
        loss = criterion(output, targets) / batch_size
        loss.backward()
        optimizer.step()

        # print(targets)
        # print(output.exp(), targets)

        return loss

    def simulate(self):
        self._restart_and_sample_chance()

        history = self.history
        search = self.batch_search
        rng = self.rng
        probs = None

        infoset = history.infoset()
        search.target(infoset)

        for i in range(self.n_simulation_iter):
            batch = search.generate_batch()

            if len(batch) == 0:
                search.step(rng)
            else:
                with torch.no_grad():
                    logits = self.model.forward(batch)
                    probs = logits.exp()
                search.step(probs, rng)

        tree = search.tree
        sigma = tree.sigma_average()

        encoding_size = encoder.encoding_size()
        max_actions = encoder.max_actions()

        infoset_encoding = torch.zeros(encoding_size)
        action_probs = torch.zeros(max_actions)

        self.encoder.encode(infoset, infoset_encoding)
        self.encoder.encode_sigma(infoset, sigma, action_probs)

        # print(infoset, infoset_encoding, action_probs)

        # avg_targeting_ratio = trainer.batch_search.avg_targeting_ratio
        # print("avg target ratio: {:.5f}".format(avg_targeting_ratio))

        if torch.rand(1).item() > play_eps:
            ap = sigma.sample_pr(infoset, rng)
            history.act(ap.a)
        else:
            actions = infoset.actions
            a_index = torch.randint(0, len(actions), (1,), dtype=torch.int).item()
            history.act(actions[a_index])

        return infoset_encoding, action_probs

    def _restart_and_sample_chance(self):
        while True:
            history = self.history
            if history.is_terminal():
                self.history = copy(self.root_history)
                self.batch_search = self.make_batch_search()
            elif history.player == oz.Chance:
                ap = history.sample_chance(rng)
                history.act(ap.a)
            else:
                break


trainer = Trainer(history=history,
                  model=model,
                  make_batch_search=make_batch_search,
                  encoder=encoder,
                  optimizer=optimizer,
                  criterion=criterion,
                  rng=rng, n_simulation_iter=n_simulation_iter)


# TODO make this more efficient
def pr_nn(infoset, action):
    enc = trainer.encoder
    d = torch.zeros(enc.encoding_size())
    enc.encode(infoset, d)
    sigma_logits = trainer.model.forward(d.unsqueeze(0))
    sigma_pr = sigma_logits.exp()
    m = enc.decode(infoset, sigma_pr.data[0])
    return m[action]


sigma_nn = oz.make_py_sigma(pr_nn)


def run_trainer_reservoir(trainer, n_iter):
    encoding_size = trainer.encoder.encoding_size()
    max_actions = trainer.encoder.max_actions()
    size = [reservoir_size, encoding_size + max_actions]
    reservoir = oz.reservoir.ExponentialReservoir(
                    sample_size=size,
                    beta_ratio=beta_ratio)

    for i in range(n_iter):
        for j in range(n_game_steps):
            infoset_encoding, action_probs = trainer.simulate()
            d = torch.cat((infoset_encoding, action_probs))
            reservoir.add(d)
            print(".", end="", flush=True)
        print()

        losses  = torch.zeros(n_train_iter)
        sample  = reservoir.sample()
        data    = sample[:,:encoding_size]
        targets = sample[:,encoding_size:]

        data_batched = data.view(-1, train_batch_size, encoding_size)
        targets_batched = targets.view(-1, train_batch_size, max_actions)
        n_batches = data_batched.size(0)
        perm = torch.randperm(n_batches)

        # TODO make minibatches
        for k in range(n_train_iter):
            x = data_batched[perm[k % n_batches]]
            y = targets_batched[perm[k % n_batches]]
            loss = trainer.train(x, y)
            losses[k] = loss

        # print(data)
        # print(targets)

        # out = trainer.model.forward(data).exp()
        # res = torch.abs(targets - out)
        # print(res.mean())
        # print(res.std())
        # print(res.max())

        ex = oz.exploitability(history, sigma_nn)
        print("ex: {:.5f}".format(ex))

        ex = oz.exploitability(history, trainer.batch_search.tree.sigma_average())
        print("avg ex: {:.5f}".format(ex))

        mean_loss = losses.mean()
        print("mean loss: {:.5f}".format(mean_loss.item()))


def run_trainer_local(trainer, n_iter):
    for i in range(n_iter):
        infoset_encoding, action_probs = trainer.simulate()
        infoset_encoding = infoset_encoding.unsqueeze(dim=0)
        action_probs = action_probs.unsqueeze(dim=0)

        if i % 50 == 0:
            ex = oz.exploitability(history, sigma_nn)
            print("ex: {:.5f}".format(ex))

        loss = trainer.train(infoset_encoding, action_probs)
        print("loss: {:.5f}".format(loss.item()))


def run_trainer_distributed(trainer):
    pass

if __name__ == "__main__":
    # run_trainer_local(trainer, 25000)
    run_trainer_reservoir(trainer, 25000)
