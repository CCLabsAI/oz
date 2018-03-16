import unittest
from .context import oz

from oz.game.flipguess import FlipGuess, Player, Action


class TestFlipGuess(unittest.TestCase):

    def test_init(self):
        h = FlipGuess()
        self.assertIsNotNone(h)

    def test_actions(self):
        h = FlipGuess()
        infoset = h.infoset()
        self.assertEqual(h.player, Player.Chance)
        self.assertEqual(infoset.actions, Action.chance_actions)
        self.assertEqual(infoset.probs, [.5, .5])
        a = Action.Tails
        h >> a
        self.assertEqual(h.player, Player.P1)
        p1_info = h.infoset()
        self.assertEqual(p1_info.actions, Action.player_actions)
        a1 = p1_info.actions[0]
        h >> a1
        self.assertEqual(h.player, Player.P2)
        p2_info = h.infoset()
        self.assertNotEqual(p1_info, p2_info)
        a2 = p2_info.actions[0]
        h >> a2
        self.assertEqual(h.utility(), 3)
