import unittest
from .context import oz


class TestExt(unittest.TestCase):

    def test_init(self):
        self.assertIsNotNone(oz._ext)

    def test_batch_segfault_regression(self):
        root = oz.make_leduk_history()
        enc = oz.LedukEncoder()
        rng = oz.Random()

        search_size = 50
        bs = oz.BatchSearch(root, enc, search_size)

        batch = bs.generate_batch()
        self.assertIsNotNone(batch)

        enc = None
        batch = bs.generate_batch()
        self.assertIsNotNone(batch)

    def test_py_sigma(self):
        h = oz.make_kuhn_history()

        def pr_callback(infoset, action):
            return 1.0 / len(infoset.actions)

        py_sigma = oz.make_py_sigma(pr_callback)

        ex = oz.exploitability(h, py_sigma)
        self.assertIsNotNone(ex)
