import unittest
from .context import oz


class TestExt(unittest.TestCase):

    def test_init(self):
        self.assertIsNotNone(oz._ext)
