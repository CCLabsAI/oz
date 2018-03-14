import unittest

from oz.game.leduk import LedukPoker, Action, Card


class TestLedukPoker(unittest.TestCase):

    def test_init(self):
        g = LedukPoker()

    def test_deal(self):
        g = LedukPoker()
        self.assertIsNotNone(g.hand[0])
        self.assertIsNotNone(g.hand[1])
        self.assertEqual(len(g.deck), 4)
        self.assertIsNone(g.board)
        self.assertEqual(g.round, 0)

    def test_bet(self):
        g = LedukPoker()
        g.act(Action.Bet)
        g.act(Action.Call)
        self.assertEqual(g.pot[0], 3)
        self.assertEqual(g.pot[1], 3)
        self.assertEqual(g.round, 1)

    def test_check(self):
        g = LedukPoker()
        g.act(Action.Check)
        g.act(Action.Call)
        self.assertEqual(g.pot[0], 1)
        self.assertEqual(g.pot[1], 1)
        self.assertEqual(g.round, 1)
        self.assertIsNotNone(g.board)

    def test_fold(self):
        g = LedukPoker()
        g.act(Action.Bet)
        g.act(Action.Fold)
        self.assertEqual(g.pot[0], 3)
        self.assertEqual(g.pot[1], 1)
        self.assertTrue(g.is_terminal())
        self.assertEqual(g.reward(), 1)

    def test_reraise(self):
        g = LedukPoker()
        g.act(Action.Bet)
        g.act(Action.Bet)
        with self.assertRaises(ValueError) as ex:
            g.act(Action.Bet)

    def test_bet_rounds(self):
        g = LedukPoker()
        g.act(Action.Bet)
        g.act(Action.Raise)
        g.act(Action.Call)
        g.act(Action.Bet)
        g.act(Action.Raise)
        g.act(Action.Call)
        self.assertEqual(g.pot[0], 13)
        self.assertEqual(g.pot[1], 13)
        self.assertTrue(g.is_terminal())

    def test_hand_rank(self):
        g = LedukPoker()
        self.assertGreater(
            g._rank_hand(card=Card.Jack, board=Card.Jack),
            g._rank_hand(card=Card.King, board=Card.Queen))
        self.assertGreater(
            g._rank_hand(card=Card.King,  board=Card.Queen),
            g._rank_hand(card=Card.Queen, board=Card.King))

    def test_reward(self):
        # P1 Win
        g = LedukPoker()
        g.act(Action.Bet)
        g.act(Action.Call)
        g.act(Action.Bet)
        g.act(Action.Call)
        g.hand[0] = Card.King
        g.hand[1] = Card.Queen
        g.board = Card.Jack
        self.assertEqual(g.reward(), 7)

        # P2 Win
        g = LedukPoker()
        g.act(Action.Bet)
        g.act(Action.Raise)
        g.act(Action.Call)
        g.act(Action.Bet)
        g.act(Action.Raise)
        g.act(Action.Call)
        g.hand[0] = Card.King
        g.hand[1] = Card.Queen
        g.board = Card.Queen
        self.assertEqual(g.reward(), -13)

        # Draw
        g = LedukPoker()
        g.act(Action.Bet)
        g.act(Action.Call)
        g.act(Action.Bet)
        g.act(Action.Call)
        g.hand[0] = Card.Queen
        g.hand[1] = Card.Queen
        g.board = Card.King
        self.assertEqual(g.reward(), 0)

    def test_legal_actions(self):
        g = LedukPoker()
        g.act(Action.Call)
        actions = g.legal_actions()
        self.assertIn(Action.Call, actions)
        self.assertIn(Action.Fold, actions)
        self.assertIn(Action.Raise, actions)

        g = LedukPoker()
        g.act(Action.Call)
        g.act(Action.Raise)
        g.act(Action.Raise)
        actions = g.legal_actions()
        self.assertNotIn(Action.Raise, actions)
