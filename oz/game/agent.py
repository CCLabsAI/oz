import random

from . import leduk


class RandomAgent():
    def choose_action(self, g):
        return random.choice(g.legal_actions())


class CallAgent():
    def choose_action(self, g):
        return leduk.Action.Call


class BiasedRandomAgent():
    def choose_action(self, g):
        r = random.random()
        if r < 0.1:
            return leduk.Action.Fold
        elif r < 0.55:
            if leduk.Action.Raise in g.legal_actions():
                return leduk.Action.Raise
            else:
                return leduk.Action.Call
        else:
            return leduk.Action.Call
