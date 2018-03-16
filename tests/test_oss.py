import unittest
from .context import oz

from enum import Enum
import random
import oz.oss as oss
from oz.game.flipguess import FlipGuess

class TerminalHistory:
    class Player(Enum):
        Chance = 0

    @property
    def current_player(self):
        return self.Player.Chance

    def is_terminal(self):
        return True

    def utility(self):
        return 1.


class TestContext:
    def __init__(self):
        self.delta = .8


class TestTree:
    pass


class UniformSigma:
    def sample_prob(self, infoset):
        actions = infoset.actions
        a = random.choice(actions)
        pr_a = 1./len(actions)
        return a, pr_a


class TestPlayoutSigma:
    def sample_prob(self, infoset):
        actions = infoset.actions
        tails = FlipGuess.Action.Tails
        left = FlipGuess.Action.Left
        if tails in actions:
            return tails, .5
        elif left in actions:
            return left, .5
        else:
            raise RuntimeError

class TestOSS(unittest.TestCase):

    def test_terminal(self):
        h = TerminalHistory()
        context = TestContext()
        tree = TestTree()
        x, l, u = oss.oss(h, context, tree, 1, 1, 1, 1, 0)
        self.assertEqual(x, 1)

    def test_playout(self):
        h = FlipGuess()
        sigma = TestPlayoutSigma()
        s = 0.1
        x_target = .5**3
        x, l, u = oss.playout(h, s, sigma)
        self.assertEqual(x, x_target)
        self.assertEqual(l, s*x_target)
        self.assertEqual(u, 3)
