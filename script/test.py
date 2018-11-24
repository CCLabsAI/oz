import sys
import random
import time
from copy import copy
import argparse
import subprocess
import os

import oz

class OOSPlayer:
    def __init__(self, history_root, n_iter, eps, delta, gamma, beta):
        self.history_root = copy(history_root)
        self.tree = oz.Tree()
        self.oos = oz.OOS()
        self.n_iter = n_iter
        self.eps = eps
        self.delta = delta
        self.gamma = gamma
        self.beta = beta

    def sample_action(self, infoset, rng):
        sigma = self.tree.sigma_average()
        ap = sigma.sample_pr(infoset, rng)
        return ap.a

    def think(self, infoset, rng):
        # self.oos.retarget()
        self.oos.search(
            self.history_root,
            self.n_iter, self.tree, rng,
            eps=self.eps, delta=self.eps, gamma=self.gamma, beta=self.beta)


class TargetedOOSPlayer(OOSPlayer):
    def __init__(self, history_root, target, **kwargs):
        super().__init__(history_root, **kwargs)
        self.target = target

    def think(self, infoset, rng):
        self.oos.search_targeted(
            self.history_root,
            self.n_iter, self.tree, rng,
            self.target, infoset,
            eps=self.eps, delta=self.eps, gamma=self.gamma)

# TODO move me
import oz.nn
import torch
import torch.nn.functional as F
import argparse


class NeuralNetPlayer:
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.encoding_size = encoder.encoding_size()

    def sample_action(self, infoset, rng):
        encoder = self.encoder
        infoset_encoded = torch.zeros(self.encoding_size)
        encoder.encode(infoset, infoset_encoded)
        logits = self.model.forward(infoset_encoded.unsqueeze(0))
        probs = F.softmax(logits, dim=1)
        ap = encoder.decode_and_sample(infoset, probs[0], rng)
        return ap.a

    def think(self, infoset, rng):
        pass


class NeuralOOSPlayer:
    def __init__(self, model, encoder, target, history_root,
                 simulation_iter, search_batch_size,
                 eps, delta, gamma, beta):
        self.model = model
        self.encoder = encoder
        self.target = target
        self.history_root = history_root
        self.simulation_iter = simulation_iter
        self.search_batch_size = search_batch_size
        self.eps = eps
        self.delta = delta
        self.gamma = gamma
        self.beta = beta
        self.eta = 1.0
        self.encoding_size = encoder.encoding_size()
        self.batch_search = self.make_batch_search()

    def make_batch_search(self):
        return oz.BatchSearch(batch_size=self.search_batch_size,
                              history=self.history_root,
                              encoder=self.encoder,
                              target=self.target,
                              eps=self.eps, delta=self.delta, gamma=self.gamma,
                              beta=self.beta, eta=self.eta)

    def sample_action(self, infoset, rng):
        sigma = self.batch_search.tree.sigma_average()
        ap = sigma.sample_pr(infoset, rng)
        return ap.a

    def think(self, infoset, rng):
        search = self.batch_search
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


def load_checkpoint_model(encoder, checkpoint_path):
    ob = torch.load(checkpoint_path)
    encoding_size = encoder.encoding_size()
    max_actions = encoder.max_actions()

    train_args = argparse.Namespace(**ob['args'])
    model = oz.nn.model_with_args(train_args,
                input_size=encoding_size,
                output_size=max_actions)
    model.load_state_dict(ob['model_state'])
    model.eval()
    return model


class UniformRandomPlayer:
    def sample_action(self, infoset, rng):
        return random.choice(infoset.actions)

    def think(self, infoset, rng):
        pass


class SequentialPlayer:
    def sample_action(self, infoset, rng):
        actions = infoset.actions
        actions.sort(key=lambda x: x.index)
        return actions[0]

    def think(self, infoset, rng):
        pass


def play_match(h, player1, player2, rng):
    h = copy(h)
    while not h.is_terminal():
        if h.player == oz.Chance:
            ap = h.sample_chance(rng)
            h.act(ap.a)

        else:
            if h.player == oz.P1:
                player = player1
            else:
                player = player2

            infoset = h.infoset()
            player.think(infoset, rng)
            a = player.sample_action(infoset, rng)

            if h.player == oz.P1:
                print(a.index, end='-', flush=True)
            else:
                print(a.index, end='/', flush=True)

            h.act(a)

    print()
    print(h)
    return h.utility(oz.P1)


def play_matches(n_matches, make_players, h, rng):
    utilities = []
    for i in range(n_matches):
        player1, player2 = make_players()
        u = play_match(h, player1, player2, rng)
        print("Results:", int(u))
        utilities.append(u)
        print()
    return utilities

def main():

    t0 = int(round(time.time() * 1000))
    parser = argparse.ArgumentParser(description="run head-to-head play tests")
    parser.add_argument("--game", help="game to play", required=True)
    parser.add_argument("--p1", help="player 1 search algorithm", required=True)
    parser.add_argument("--p2", help="player 2 search algorithm", required=True)
    parser.add_argument("--iter1", type=int,
                        help="player 1 thinking iterations",
                        default=1000)
    parser.add_argument("--iter2", type=int,
                        help="player 2 thinking iterations",
                        default=1000)
    parser.add_argument("--matches", type=int,
                        help="number of matches to play",
                        default=80)
    parser.add_argument("--goofcards", type=int,
                        help="number of cards for II Goofspiel",
                        default=6)
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
    parser.add_argument("--checkpoint_path1",
                        help="player 1 nn checkpoint",
                        default=None)
    parser.add_argument("--checkpoint_path2",
                        help="player 2 nn checkpoint",
                        default=None)
    parser.add_argument("--search_batch_size", type=int,
                        help="number of concurrent searches when using NN",
                        default=20)

    args = parser.parse_args()
    if 'GIT_HASH' in os.environ:
        label = os.environ['GIT_HASH']
    else:
        try:
            label = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            label = label.decode('ascii').strip()
        except CalledProcessError:
            label = 'unknown-git-revision'
    print(label)

    history = None
    target = None

    if args.game == 'leduk' or args.game == 'leduk_poker':
        history = oz.make_leduk_history()
        encoder = oz.make_leduk_encoder()
        target  = oz.make_leduk_target()
    elif args.game == 'goofspiel' or args.game == 'goofspiel2':
        history = oz.make_goofspiel2_history(args.goofcards)
        encoder = oz.make_goofspiel2_encoder(args.goofcards)
        target  = oz.make_goofspiel2_target()
    else:
        print('error: unknown game: {}'.format(args.game), file=sys.stderr)
        exit(1)

    def make_player(algo, n_iter, checkpoint_path):
        if algo == 'random':
            return UniformRandomPlayer()

        elif algo == 'oos':
            return OOSPlayer(history,
                             n_iter=n_iter,
                             eps=args.eps,
                             delta=args.delta,
                             gamma=args.gamma,
                             beta=args.beta)

        elif algo == 'oos_targeted':
            return TargetedOOSPlayer(history, target,
                                     n_iter=n_iter,
                                     eps=args.eps,
                                     delta=args.delta,
                                     gamma=args.gamma,
                                     beta=args.beta)
        elif algo == 'nn':
            if checkpoint_path is None:
                print('error: missing checkpoint path', file=sys.stderr)
                exit(1)
            model = load_checkpoint_model(encoder, checkpoint_path)
            return NeuralNetPlayer(model=model, encoder=encoder)

        elif algo == 'nn_oos':
            if checkpoint_path is None:
                print('error: missing checkpoint path', file=sys.stderr)
                exit(1)
            model = load_checkpoint_model(encoder, checkpoint_path)
            return NeuralOOSPlayer(
                model=model,
                encoder=encoder,
                target=target,
                history_root=history,
                simulation_iter=n_iter,
                search_batch_size=args.search_batch_size,
                eps=args.eps, delta=args.delta, gamma=args.gamma, beta=args.beta)

        else:
            print('error: unknown search algorithm: {}'.format(algo), file=sys.stderr)
            exit(1)


    def make_players():
        player1 = make_player(args.p1, args.iter1, args.checkpoint_path1)
        player2 = make_player(args.p2, args.iter2, args.checkpoint_path2)
        return player1, player2


    rng = oz.Random()
    utilities = play_matches(args.matches, make_players, history, rng)
    t1 = int(round(time.time() * 1000))

    print("N ", args.matches)
    print("Cards number : ", args.goofcards)
    print("Players", args.p1, args.p2)
    print("Iters", args.iter1, args.iter2)
    print("Checkpoints", args.checkpoint_path1, args.checkpoint_path2)
    print("beta:", args.beta)
    print("delta:", args.delta)
    print("epsilon:", args.eps)
    print("gamma:", args.gamma)
    print("Execution time", t1-t0)








    # print(utilities)
    # print(sum(utilities)/len(utilities))

if __name__ == "__main__":
    # execute only if run as a script
    main()
