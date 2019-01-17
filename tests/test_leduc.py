import unittest
# from .context import oz

from oz.game.leduc import LeducPoker, Action, Card


class TestLeducPoker(unittest.TestCase):

    @staticmethod
    def _leduc_after_deal():
        g = LeducPoker()
        g.act(g.ChanceAction.J1)
        g.act(g.ChanceAction.Q2)
        return g

    def test_init(self):
        g = LeducPoker()
        self.assertIsNotNone(g)

    def test_deal(self):
        g = LeducPoker()
        self.assertEqual(g.player, g.Player.Chance)
        self.assertEqual(g.infoset().actions, g.ChanceAction.P1_deal)
        g.act(g.ChanceAction.J1)
        self.assertEqual(g.infoset().actions, g.ChanceAction.P2_deal)
        g.act(g.ChanceAction.Q2)
        self.assertEqual(g.player, g.Player.P1)
        g.act(g.Action.Bet)
        self.assertEqual(g.player, g.Player.P2)
        g.act(g.Action.Call)
        self.assertEqual(g.player, g.Player.Chance)

    def test_bet(self):
        g = self._leduc_after_deal()
        g.act(Action.Bet)
        g.act(Action.Call)
        self.assertEqual(g.pot[0], 3)
        self.assertEqual(g.pot[1], 3)
        self.assertEqual(g.round, 1)

    def test_check(self):
        g = self._leduc_after_deal()
        g.act(Action.Check)
        g.act(Action.Call)
        self.assertEqual(g.pot[0], 1)
        self.assertEqual(g.pot[1], 1)
        self.assertEqual(g.round, 1)
        self.assertIsNotNone(g.board)

    def test_fold(self):
        g = self._leduc_after_deal()
        g.act(Action.Bet)
        g.act(Action.Fold)
        self.assertEqual(g.pot[0], 3)
        self.assertEqual(g.pot[1], 1)
        self.assertTrue(g.is_terminal())
        self.assertEqual(g.utility(), 1)

    def test_reraise(self):
        g = self._leduc_after_deal()
        g.act(Action.Bet)
        g.act(Action.Bet)
        with self.assertRaises(ValueError):
            g.act(Action.Bet)

    def test_bet_rounds(self):
        g = self._leduc_after_deal()
        g.act(Action.Bet)
        g.act(Action.Raise)
        g.act(Action.Call)
        self.assertEqual(g.player, g.Player.Chance)
        g.act(g.ChanceAction.K)
        g.act(Action.Bet)
        g.act(Action.Raise)
        g.act(Action.Call)
        self.assertEqual(g.pot[0], 13)
        self.assertEqual(g.pot[1], 13)
        self.assertTrue(g.is_terminal())

    def test_hand_rank(self):
        g = LeducPoker()
        self.assertGreater(
            g._rank_hand(card=Card.Jack, board=Card.Jack),
            g._rank_hand(card=Card.King, board=Card.Queen))
        self.assertGreater(
            g._rank_hand(card=Card.King,  board=Card.Queen),
            g._rank_hand(card=Card.Queen, board=Card.King))

    def test_reward(self):
        # P1 Win
        g = LeducPoker()
        g.act(g.ChanceAction.K1)
        g.act(g.ChanceAction.Q2)
        g.act(g.Action.Bet)
        g.act(g.Action.Call)
        g.act(g.ChanceAction.J)
        g.act(g.Action.Bet)
        g.act(g.Action.Call)
        g.hand[0] = Card.King
        g.hand[1] = Card.Queen
        g.board = Card.Jack
        self.assertEqual(g.utility(), 7)

        # P2 Win
        g = LeducPoker()
        g.act(g.ChanceAction.K1)
        g.act(g.ChanceAction.Q2)
        g.act(Action.Bet)
        g.act(Action.Raise)
        g.act(Action.Call)
        g.act(g.ChanceAction.Q)
        g.act(Action.Bet)
        g.act(Action.Raise)
        g.act(Action.Call)
        g.hand[0] = Card.King
        g.hand[1] = Card.Queen
        g.board = Card.Queen
        self.assertEqual(g.utility(), -13)

        # Draw
        g = LeducPoker()
        g.act(g.ChanceAction.Q1)
        g.act(g.ChanceAction.Q2)
        g.act(Action.Bet)
        g.act(Action.Call)
        g.act(g.ChanceAction.K)
        g.act(Action.Bet)
        g.act(Action.Call)
        g.hand[0] = Card.Queen
        g.hand[1] = Card.Queen
        g.board = Card.King
        self.assertEqual(g.utility(), 0)

    def test_legal_actions(self):
        g = self._leduc_after_deal()
        g.act(Action.Call)
        actions = g.infoset().actions
        self.assertIn(Action.Call, actions)
        self.assertIn(Action.Fold, actions)
        self.assertIn(Action.Raise, actions)

        g = self._leduc_after_deal()
        g.act(Action.Call)
        g.act(Action.Raise)
        g.act(Action.Raise)
        actions = g.infoset().actions
        self.assertNotIn(Action.Raise, actions)
