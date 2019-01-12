import random

from . import leduc


class RandomAgent():
    def choose_action(self, g):
        return random.choice(g.legal_actions())


class CallAgent():
    def choose_action(self, g):
        return leduc.Action.Call


class BiasedRandomAgent():
    def choose_action(self, g):
        r = random.random()
        if r < 0.1:
            return leduc.Action.Fold
        elif r < 0.55:
            if leduc.Action.Raise in g.legal_actions():
                return leduc.Action.Raise
            else:
                return leduc.Action.Call
        else:
            return leduc.Action.Call
