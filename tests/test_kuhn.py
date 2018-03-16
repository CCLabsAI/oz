import unittest
from .context import oz

from oz.game.kuhn import KuhnPoker, Player, Action, ChanceAction, Card


class TestKuhnPoker(unittest.TestCase):

    def test_init(self):
        h = KuhnPoker()
        self.assertIsNotNone(h)

    def test_deal(self):
        h = KuhnPoker()
        h >> ChanceAction.KQ
        self.assertEqual(h.hand, [Card.King, Card.Queen])

    def test_play(self):
        h = KuhnPoker()

        h >> ChanceAction.KQ
        self.assertEqual(h.pot, [1, 1])
        self.assertEqual(h.player, Player.P1)

        h >> Action.Bet
        self.assertEqual(h.pot, [2, 1])
        self.assertEqual(h.player, Player.P2)

        h >> Action.Pass
        self.assertEqual(h.pot, [2, 1])
        self.assertTrue(h.is_terminal())
        self.assertEqual(h.utility(), 1)

        h = KuhnPoker()
        h >> ChanceAction.QK
        h >> Action.Pass
        self.assertEqual(h.player, Player.P2)
        h >> Action.Bet
        self.assertFalse(h.is_terminal())
        h >> Action.Bet
        self.assertTrue(h.is_terminal())
        self.assertEqual(h.utility(), -2)

        h = KuhnPoker()
        h >> ChanceAction.QJ
        h >> Action.Bet
        h >> Action.Bet
        self.assertTrue(h.is_terminal())
        self.assertEqual(h.utility(), 2)
