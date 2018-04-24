import random
from copy import copy

import oz


class OOSPlayer:
    def __init__(self, history_root, n_iter=10000):
        self.history_root = copy(history_root)
        self.tree = oz.Tree()
        self.oos = oz.OOS()
        self.n_iter = n_iter

    def sample_action(self, infoset, rng):
        sigma = self.tree.sigma_average()
        ap = sigma.sample_pr(infoset, rng)
        return ap.a

    def think(self, infoset, rng):
        self.oos.retarget()
        self.oos.search(
            self.history_root,
            self.n_iter, self.tree, rng,
            eps=0.4, delta=0.9, gamma=0.01)


class TargetedOOSPlayer(OOSPlayer):
    def __init__(self, history_root, target, n_iter=10000):
        super(TargetedOOSPlayer, self).__init__(history_root, n_iter=n_iter)
        self.target = target

    def think(self, infoset, rng):
        self.oos.retarget()
        self.oos.search_targeted(
            self.history_root,
            self.n_iter, self.tree, rng,
            self.target, infoset,
            eps=0.4, delta=0.9, gamma=0.01)


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
            h.act(a)
            print('.', end='', flush=True)

    return h.utility(oz.P1)


def play_matches(n_matches, make_players, h, rng):
    utilities = []
    for i in range(n_matches):
        player1, player2 = make_players()
        u = play_match(h, player1, player2, rng)
        utilities.append(u)
        print()
        print(u)
    return utilities


h = oz.make_goofspiel2_history(6)
t = oz.make_goofspiel2_target()


def make_players():
    # player1 = OOSPlayer(h)
    player1 = TargetedOOSPlayer(h, t, n_iter=10000)

    player2 = UniformRandomPlayer()
    # player2 = SequentialPlayer()
    # player2 = OOSPlayer(h)

    return player1, player2


rng = oz.Random()
utilities = play_matches(10, make_players, h, rng)

print(utilities)
print(sum(utilities)/len(utilities))
