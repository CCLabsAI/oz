import oz
import oz.reservoir
import numpy as np
import torch
import torch.nn.functional as F

from copy import copy

Nan = float('nan')
NInf_tensor = torch.tensor(float('-inf'))
Zero_tensor = torch.tensor(0.0)

class Trainer:
    def __init__(self, history, make_batch_search, encoder,
                 model, optimizer, criterion,
                 simulation_iter, play_eps, rng):
        self.root_history = copy(history)
        self.history = copy(history)
        self.make_batch_search = make_batch_search
        self.batch_search = make_batch_search()
        self.encoder = encoder
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.simulation_iter = simulation_iter
        self.play_eps = play_eps
        self.rng = rng

    def train(self, data, targets):
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        batch_size = data.size(0)

        mask = torch.isnan(targets)

        optimizer.zero_grad()
        logits = model(data);

        # NB a nan target means that action is illegal
        # we manually replace the logit for illegal actions with
        # negative infinity
        logits = torch.where(mask, NInf_tensor, logits)
        logits = F.log_softmax(logits, dim=1)

        # NB it seems that because of a (p > 0) check inside the
        # implementation of kl_div means that a nan probability
        # simply means a 0 loss
        loss = criterion(logits, targets) / batch_size

        # Here is an alternative version that zeros out the probability
        # of illegal actions:
        # targets_zeroed = torch.where(mask, Zero_tensor, targets)
        # loss = criterion(logits, targets_zeroed) / batch_size

        loss.backward()
        optimizer.step()

        return loss

    def simulate(self):
        self._restart_and_sample_chance()

        history = self.history
        search = self.batch_search
        encoder = self.encoder
        rng = self.rng
        probs = None

        infoset = history.infoset()
        search.target(infoset)

        for i in range(self.simulation_iter):
            batch = search.generate_batch()

            if len(batch) == 0:
                search.step(rng)
            else:
                with torch.no_grad():
                    logits = self.model.forward(batch)
                    probs = F.softmax(logits, dim=1)
                search.step(probs, rng)

        tree = search.tree
        sigma = tree.sigma_average()

        encoding_size = encoder.encoding_size()
        max_actions = encoder.max_actions()

        infoset_encoding = torch.zeros(encoding_size)
        action_probs = torch.full((max_actions,), Nan)

        self.encoder.encode(infoset, infoset_encoding)
        self.encoder.encode_sigma(infoset, sigma, action_probs)

        if torch.rand(1).item() > self.play_eps:
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
                sigma = self.batch_search.tree.sigma_average()

                self.history = copy(self.root_history)
                self.batch_search = self.make_batch_search()
            elif history.player == oz.Chance:
                ap = history.sample_chance(self.rng)
                history.act(ap.a)
            else:
                break


def run_trainer(trainer, n_iter):
    for i in range(n_iter):
        infoset_encoding, action_probs = trainer.simulate()
        infoset_encoding = infoset_encoding.unsqueeze(dim=0)
        action_probs = action_probs.unsqueeze(dim=0)

        loss = trainer.train(infoset_encoding, action_probs)
        print("loss: {:.5f}".format(loss.item()))


interrupted = False


def run_trainer_reservoir(trainer, args, start_iteration=0, iter_callback=None):
    global interrupted

    train_batch_size = args.train_batch_size
    train_game_ply = args.train_game_ply
    train_steps = args.train_steps
    train_iter = args.train_iter

    reservoir_beta_ratio = args.reservoir_beta_ratio
    reservoir_size = args.reservoir_size

    encoding_size = trainer.encoder.encoding_size()
    max_actions = trainer.encoder.max_actions()
    size = [reservoir_size, encoding_size + max_actions]
    reservoir = oz.reservoir.ExponentialReservoir(
                    sample_size=size,
                    beta_ratio=reservoir_beta_ratio)

    print("reservoir p_k:", reservoir.p_k)

    iteration_n = start_iteration
    while iteration_n < train_iter:
        print("starting iteration: ", iteration_n, flush=True)
        for j in range(train_game_ply):
            if interrupted:
                break
            infoset_encoding, action_probs = trainer.simulate()
            d = torch.cat((infoset_encoding, action_probs))
            reservoir.add(d)
            if args.progress:
                print(".", end="", flush=True)
        if args.progress:
            print(flush=True)

        if interrupted:
            if iter_callback:
                iter_callback(
                    iteration_n=iteration_n,
                    args=args,
                    interrupted=True,
                    trainer=trainer,
                    losses=None)
            break

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
