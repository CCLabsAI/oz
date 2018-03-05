import unittest
from .context import oz

from oz.game.leduk import LedukPoker

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
        g.act('bet_or_raise')
        g.act('check_or_call')
        self.assertEqual(g.pot[0], 3)
        self.assertEqual(g.pot[1], 3)
        self.assertEqual(g.round, 1)

    def test_check(self):
        g = LedukPoker()
        g.act('check_or_call')
        g.act('check_or_call')
        self.assertEqual(g.pot[0], 1)
        self.assertEqual(g.pot[1], 1)
        self.assertEqual(g.round, 1)
        self.assertIsNotNone(g.board)

    def test_fold(self):
        g = LedukPoker()
        g.act('bet_or_raise')
        g.act('fold')
        self.assertEqual(g.pot[0], 3)
        self.assertEqual(g.pot[1], 1)
        self.assertTrue(g.is_terminal())

    def test_reraise(self):
        g = LedukPoker()
        g.act('bet_or_raise')
        g.act('bet_or_raise')
        with self.assertRaises(ValueError) as ex:
            g.act('bet_or_raise')

    def test_bet_rounds(self):
        g = LedukPoker()
        g.act('bet_or_raise')
        g.act('bet_or_raise')
        g.act('check_or_call')
        g.act('bet_or_raise')
        g.act('bet_or_raise')
        g.act('check_or_call')
        self.assertEqual(g.pot[0], 13)
        self.assertEqual(g.pot[1], 13)
        self.assertTrue(g.is_terminal())

    def test_hand_rank(self):
        g = LedukPoker()
        self.assertGreater(
            g._rank_hand(card=g.JACK, board=g.JACK),
            g._rank_hand(card=g.KING, board=g.QUEEN))
        self.assertGreater(
            g._rank_hand(card=g.KING,  board=g.QUEEN),
            g._rank_hand(card=g.QUEEN, board=g.KING))

    def test_reward(self):
        # P2 Win
        g = LedukPoker()
        g.act('bet_or_raise')
        g.act('bet_or_raise')
        g.act('check_or_call')
        g.act('bet_or_raise')
        g.act('bet_or_raise')
        g.act('check_or_call')
        g.hand[0] = g.KING
        g.hand[1] = g.QUEEN
        g.board = g.QUEEN
        self.assertEqual(g.reward(), -13)

        # P1 Win
        g = LedukPoker()
        g.act('bet_or_raise')
        g.act('check_or_call')
        g.act('bet_or_raise')
        g.act('check_or_call')
        g.hand[0] = g.KING
        g.hand[1] = g.QUEEN
        g.board = g.JACK
        self.assertEqual(g.reward(), 7)

        # Draw
        g = LedukPoker()
        g.act('bet_or_raise')
        g.act('check_or_call')
        g.act('bet_or_raise')
        g.act('check_or_call')
        g.hand[0] = g.QUEEN
        g.hand[1] = g.QUEEN
        g.board = g.KING
        self.assertEqual(g.reward(), 0)
