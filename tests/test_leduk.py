import unittest
from .context import oz

from oz.game.leduk import LedukPoker

class TestLedukPoker(unittest.TestCase):

    def test_init(self):
        g = LedukPoker()
