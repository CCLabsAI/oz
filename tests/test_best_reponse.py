import unittest
# from .context import oz

from oz.game.flipguess import FlipGuess
from oz.game.kuhn import KuhnPoker

from oz import best_response


class SigmaUniform:
    def pr(self, infoset, a):
        actions = infoset.actions
        pr_a = 1./len(actions)
        return pr_a


class SigmaFlip:
    def pr(self, infoset, a):
        if infoset.player == FlipGuess.Player.P1:
            return 0.5
        elif infoset.player == FlipGuess.Player.P2:
            if a == FlipGuess.Action.Right:
                return 2.0/3.0
            elif a == FlipGuess.Action.Left:
                return 1.0/3.0
            else:
                raise ValueError()
        else:
            return 1./len(infoset.actions)


class SigmaKuhn:
    def pr(self, infoset, a):
        if infoset.player == KuhnPoker.Player.P1 or \
           infoset.player == KuhnPoker.Player.P2:
            if a == KuhnPoker.Action.Bet:
                return 1.0
            elif a == KuhnPoker.Action.Pass:
                return 0.0
        else:
            return 1./len(infoset.actions)


class TestBestResponse(unittest.TestCase):

    def test_gebr_flip_p1(self):
        h = FlipGuess()
        sigma_uniform = SigmaUniform()
        v1 = best_response.gebr(h, h.Player.P1, sigma_uniform)
        self.assertEqual(v1, 1.25)

    def test_gebr_flip_p2(self):
        h = FlipGuess()
        sigma_uniform = SigmaUniform()
        v2 = best_response.gebr(h, h.Player.P2, sigma_uniform)
        self.assertEqual(-v2, 1)

    def test_gebr_kuhn(self):
        h = KuhnPoker()
        sigma = SigmaKuhn()
        v2 = best_response.gebr(h, h.Player.P2, sigma)
        self.assertAlmostEqual(-v2, (1./3)*1 + (1./3)*-2)

    def test_exploitability(self):
        h = FlipGuess()
        sigma_uniform = SigmaUniform()
        ex = best_response.exploitability(h, sigma_uniform)
        self.assertEqual(ex, 0.25)

        sigma_flip = SigmaFlip()
        ex = best_response.exploitability(h, sigma_flip)
        self.assertEqual(ex, 0)
