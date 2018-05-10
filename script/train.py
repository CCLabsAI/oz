import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data

import oz
import oz.reservoir

import os
import argparse
from copy import copy
import sys
import signal

interrupted = False

def sigint_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, sigint_handler)


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
        batch_size = data.shape[0]

        optimizer.zero_grad()
        output = model(data);
        loss = criterion(output, targets) / batch_size
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
                ex = oz.exploitability(self.root_history, sigma)
                print("game ex: {:.5f}".format(ex))

                self.history = copy(self.root_history)
                self.batch_search = self.make_batch_search()
            elif history.player == oz.Chance:
                ap = history.sample_chance(self.rng)
                history.act(ap.a)
            else:
                break

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
        print("starting iteration: ", iteration_n)
        for j in range(train_game_ply):
            if interrupted:
                break
            infoset_encoding, action_probs = trainer.simulate()
            d = torch.cat((infoset_encoding, action_probs))
            reservoir.add(d)
            print(".", end="", flush=True)
        print()

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

        data_batched = data.view(-1, train_batch_size, encoding_size)
        targets_batched = targets.view(-1, train_batch_size, max_actions)
        n_batches = data_batched.size(0)
        perm = torch.randperm(n_batches)

        for k in range(train_steps):
            x = data_batched[perm[k % n_batches]]
            y = targets_batched[perm[k % n_batches]]
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

def main():
    parser = argparse.ArgumentParser(description="train NN with OmegaZero")
    parser.add_argument("--game", help="game to play", required=True)
    parser.add_argument("--nn_arch",
                        help="nn architecture",
                        default="mlp")
    parser.add_argument("--hidden_size", type=int,
                        help="mlp hidden layer size",
                        default=64)
    parser.add_argument("--checkpoint_dir",
                        help="checkpoint directory")
    parser.add_argument("--checkpoint_interval", type=int,
                        help="number of iterations between checkpoints",
                        default=25)
    parser.add_argument("--goofcards", type=int,
                        help="number of cards for II Goofspiel",
                        default=6)
    parser.add_argument("--learning_rate", type=float,
                        help="learning rate",
                        default=0.01)
    parser.add_argument("--eps", type=float,
                        help="exploration factor",
                        default=0.4)
    parser.add_argument("--delta", type=float,
                        help="targeting factor",
                        default=0.9)
    parser.add_argument("--gamma", type=float,
                        help="opponent error factor",
                        default=0.01)
    parser.add_argument("--beta", type=float,
                        help="opponent error factor",
                        default=0.99)
    parser.add_argument("--eta", type=float,
                        help="currently unused",
                        default=1.0)
    parser.add_argument("--search_batch_size", type=int,
                        help="search batch size",
                        default=20)
    parser.add_argument("--play_eps", type=float,
                        help="playout eps",
                        default=0.2)
    parser.add_argument("--simulation_iter", type=int,
                        help="simulation iterations",
                        default=5000)
    parser.add_argument("--reservoir_size", type=int,
                        help="reservoir size",
                        default=2**17)
    parser.add_argument("--reservoir_beta_ratio", type=float,
                        help="simulation iterations",
                        default=2.0)
    parser.add_argument("--train_game_ply", type=int,
                        help="number of plies between nn training",
                        default=128)
    parser.add_argument("--train_batch_size", type=int,
                        help="nn training batch size",
                        default=128)
    parser.add_argument("--train_steps", type=int,
                        help="nn training steps per iteration",
                        default=256)
    parser.add_argument("--train_iter", type=int,
                        help="nn training iterations",
                        default=10000)

    args = parser.parse_args()

    if os.path.exists(args.checkpoint_dir):
        files = [f for f in os.listdir(args.checkpoint_dir)
                  if not f.startswith('.')]

        if files:
            files.sort()
            latest_checkpoint_name = files[-1]
            checkpoint_path = os.path.join(args.checkpoint_dir,
                                           latest_checkpoint_name)
            print("loading checkpoint: {}".format(checkpoint_path))
            ob = torch.load(checkpoint_path)
            checkpoint_args = argparse.Namespace(**ob['args'])
            # TODO print warning if loaded args differ
            run(checkpoint_args, checkpoint_data=ob)
        else:
            run(args)

    else:
        os.makedirs(args.checkpoint_dir)
        run(args)

def save_checkpoint(iteration_n, trainer, args):
    ob = {
        'iteration_n': iteration_n,
        'args': vars(args),
        'model_state': trainer.model.state_dict(),
        'optimizer_state': trainer.optimizer.state_dict()
    }

    dir_path = args.checkpoint_dir
    checkpoint_name = "checkpoint-{:06d}.pth".format(iteration_n)
    full_path = os.path.join(dir_path, checkpoint_name)
    torch.save(ob, full_path)
    print('saved checkpoint: {}'.format(full_path))

def run(args, checkpoint_data=None):
    game = args.game
    if game == 'leduk' or game == 'leduk_poker':
        history = oz.make_leduk_history()
        encoder = oz.make_leduk_encoder()
        target  = oz.make_leduk_target()
    elif game == 'goofspiel' or game == 'goofspiel2':
        n_cards = args.goofcards
        history = oz.make_goofspiel2_history(n_cards)
        encoder = oz.make_goofspiel2_encoder()
        target  = oz.make_goofspiel2_target(n_cards)

    rng = oz.Random()

    model = Net(input_size=encoder.encoding_size(),
                hidden_size=args.hidden_size,
                output_size=encoder.max_actions())

    def make_batch_search():
        return oz.BatchSearch(batch_size=args.search_batch_size,
                              history=history,
                              encoder=encoder,
                              target=target,
                              eps=args.eps, delta=args.delta, gamma=args.gamma,
                              beta=args.beta, eta=args.eta)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(size_average=False)

    trainer = Trainer(history=history,
                      model=model,
                      make_batch_search=make_batch_search,
                      encoder=encoder,
                      optimizer=optimizer,
                      criterion=criterion,
                      simulation_iter=args.simulation_iter,
                      play_eps=args.play_eps,
                      rng=rng)

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

    def iter_callback(iteration_n, interrupted, trainer, args, losses):
        if not interrupted:
            ex = oz.exploitability(trainer.root_history, sigma_nn)
            print("ex: {:.5f}".format(ex))

            mean_loss = losses.mean()
            print("mean loss: {:.5f}".format(mean_loss.item()))

            if iteration_n % args.checkpoint_interval == 0:
                save_checkpoint(iteration_n=iteration_n,
                                trainer=trainer,
                                args=args)

        else:
            save_checkpoint(iteration_n=iteration_n,
                            trainer=trainer,
                            args=args)

    # load params from checkpoint
    if checkpoint_data:
        start_iteration = checkpoint_data['iteration_n']
        model.load_state_dict(checkpoint_data['model_state'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state'])
    else:
        start_iteration = 0

    run_trainer_reservoir(trainer, args,
        start_iteration=start_iteration,
        iter_callback=iter_callback)

if __name__ == "__main__":
    main()
