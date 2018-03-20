import unittest
from .context import oz

from enum import Enum
from copy import copy
import oz.oss as oss
from oz.game.flipguess import FlipGuess


class TerminalHistory:
    class Player(Enum):
        Chance = 0

    def __init__(self):
        self.player = self.Player.Chance

    def is_terminal(self):
        return True

    def utility(self, player):
        return 1


class TestContext:
    def __init__(self):
        self.delta = .8


class TestTree:
    pass


class TestPlayoutSigma:
    def sample_pr(self, infoset):
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

    def test_flipguess(self):
        h = FlipGuess()
        tree = oss.Tree()
        context = oss.Context()

        for i in range(10000):
            oss.oss(copy(h), context, tree, 1, 1, 1, 1, h.Player.P1)
            oss.oss(copy(h), context, tree, 1, 1, 1, 1, h.Player.P2)

        self.assertEqual(len(tree.nodes), 2)

        node = tree.nodes[FlipGuess.PlayerInfoset(FlipGuess.Player.P2)]
        nl = node.average_strategy[FlipGuess.Action.Left]
        nr = node.average_strategy[FlipGuess.Action.Right]
        self.assertAlmostEqual(nl / (nl + nr), 1./3, places=2)
