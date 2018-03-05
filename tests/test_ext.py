import unittest
from .context import oz

class TestLedukPoker(unittest.TestCase):

    def test_init(self):
        self.assertIsNotNone(oz.__version__)
