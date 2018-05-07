import sys
import random
import time
from copy import copy
import argparse
import subprocess

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

    args = parser.parse_args()
    #label = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    #print(label)

    history = None
    target = None

    if args.game == 'leduk' or args.game == 'leduk_poker':
        history = oz.make_leduk_history()
        target = oz.make_leduk_target()
    elif args.game == 'goofspiel' or args.game == 'goofspiel2':
        history = oz.make_goofspiel2_history(args.goofcards)
        target = oz.make_goofspiel2_target()
    else:
        print('error: unknown game: {}'.format(args.game), file=sys.stderr)
        exit(1)

    def make_player(algo, n_iter):
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

        else:
            print('error: unknown search algorithm: {}'.format(algo),
                  file=sys.stderr)
            exit(1)


    def make_players():
        player1 = make_player(args.p1, args.iter1)
        player2 = make_player(args.p2, args.iter2)
        return player1, player2


    rng = oz.Random()
    utilities = play_matches(args.matches, make_players, history, rng)
    t1 = int(round(time.time() * 1000))

    print("N ", args.matches)
    print("Cards number : ", args.goofcards)
    print("Players", args.p1, args.p2)
    print("Iters ", args.iter1, args.iter2)
    print("beta:", args.beta)
    print("delta:", args.delta)
    print("epsilon:", args.eps)
    print("gamma:", args.gamma)
    print("Execution time : ", t1-t0)








    # print(utilities)
    # print(sum(utilities)/len(utilities))

if __name__ == "__main__":
    # execute only if run as a script
    main()
