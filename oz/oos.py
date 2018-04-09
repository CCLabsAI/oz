from copy import copy
from collections import defaultdict

import numpy.random
import numpy as np

def sample(h, context, infoset, sigma, s1, s2, i):
    # FIXME implement targeting
    # a, rho1, rho2 = sigma.sample_targeted(context, infoset, eps)
    # return a, rho1*s1, rho2*s2
    player = h.player
    if player == i:
        eps = context.eps
        a, pr_a = sigma.sample_eps(infoset, eps)
        return a, pr_a*s1, pr_a*s2
    else:
        a, pr_a = sigma.sample_pr(infoset)
        return a, pr_a*s1, pr_a*s2


def sample_chance(h, context):
    rng = context.rng
    infoset = h.infoset()
    actions = infoset.actions
    probs = infoset.probs
    i = rng.choice(len(actions), p=probs)
    a = actions[i]
    pr_a = probs[i]
    return a, pr_a, pr_a


def playout(h, s, sigma):
    # FIXME chance player
    x = 1.
    while not h.is_terminal():
        infoset = h.infoset()
        a, pr_a = sigma.sample_pr(infoset)
        h = h >> a
        x = pr_a*x
    u = h.utility()
    return x, s*x, u


def _regret_to_pr(regrets):
    total = sum(v for v in regrets.values() if v > 0)

    probs = {}
    if total > 0:
        for a, r in regrets.items():
            if r > 0:
                probs[a] = r / total
            else:
                probs[a] = 0
    else:
        p = 1./len(regrets)
        for a in regrets:
            probs[a] = p

    return probs


class SigmaUniform:
    def __init__(self, rng):
        self._rng = rng

    def pr(self, infoset, a):
        actions = infoset.actions
        return 1./len(actions)

    def sample_pr(self, infoset):
        rng = self._rng
        actions = infoset.actions
        a = rng.choice(actions)
        pr_a = 1./len(actions)
        return a, pr_a

    def sample_eps(self, infoset, eps):
        return self.sample_pr(infoset)


class SigmaRegretMatching:
    def __init__(self, regrets, rng):
        self.regrets = regrets
        self._rng = rng
        pr = _regret_to_pr(regrets)
        actions = list(pr.keys())
        actions.sort(key=lambda x: x.name)
        probs = [pr[a] for a in actions]
        self._pr = pr
        self._actions = actions
        self._probs = probs

    def pr(self, infoset, a):
        return self._pr[a]

    def sample_pr(self, infoset):
        rng = self._rng
        a = rng.choice(self._actions, p=self._probs)
        return a, self._pr[a]

    def sample_eps(self, infoset, eps):
        rng = self._rng
        if rng.rand() > eps:
            a = rng.choice(self._actions, p=self._probs)
        else:
            a = rng.choice(self._actions)
        pr_a_eps = eps*(1./len(self._actions)) + (1-eps)*self._pr[a]
        return a, pr_a_eps


class SigmaAverageStrategy:
    def __init__(self, tree, rng):
        self.tree = tree
        self._rng = rng

    def pr(self, infoset, a, uniform_fallback=True):
        rng = self._rng
        if uniform_fallback:
            node = self.tree.nodes.get(infoset)
        else:
            node = self.tree.nodes[infoset]

        if node is not None:
            total = sum(node.average_strategy.values())
            if total == 0.0:
                actions = infoset.actions
                return 1. / len(actions)
            return float(node.average_strategy[a]) / total
        else:
            actions = infoset.actions
            return 1. / len(actions)

    def sample_pr(self, infoset, uniform_fallback=True):
        rng = self._rng
        if uniform_fallback:
            node = self.tree.nodes.get(infoset)
        else:
            node = self.tree.nodes[infoset]

        if node is not None:
            average_strategy = node.average_strategy
            actions = average_strategy.keys()
            values = average_strategy.values()
            total = sum(values)
            probs = [v / total for v in values]
            a = rng.choice(actions, p=probs)
            pr_a = average_strategy[a] / total
            return a, pr_a
        else:
            actions = infoset.actions
            a = rng.choice(actions)
            pr_a = 1. / len(actions)
            return a, pr_a

    def sample_eps(self, infoset, eps):
        rng = self._rng
        node = self.tree.nodes[infoset]
        average_strategy = node.average_strategy
        actions = average_strategy.keys()
        values = average_strategy.values()
        total = sum(values)
        if rng.rand() > eps:
            probs = [v / total for v in values]
            a = rng.choice(actions, p=probs)
        else:
            a = rng.choice(actions)
        pr_a = average_strategy[a] / total
        pr_a_eps = eps*(1./len(actions)) + (1-eps)*pr_a
        return a, pr_a_eps


class Tree:
    class Node:
        def __init__(self):
            self.regrets = defaultdict(float)
            self.average_strategy = defaultdict(float)

        def sigma_regret_matching(self, rng):
            # TODO Make this less ugly
            if len(self.regrets) > 0:
                return SigmaRegretMatching(self.regrets, rng=rng)
            else:
                return SigmaUniform(rng=rng)

        def update_regret(self, a, r):
            self.regrets[a] += r

        def update_average_strategy(self, a, s):
            self.average_strategy[a] += s

    def __init__(self):
        self.nodes = {}

    def lookup(self, infoset):
        if infoset in self.nodes:
            return self.nodes[infoset], False
        else:
            node = self.Node()
            self.nodes[infoset] = node
            return node, True

    def sigma_average_strategy(self, rng):
        return SigmaAverageStrategy(self, rng)


class Context:
    def __init__(self, eps=0.5, sigma=0.4, seed=None):
        self.rng = numpy.random.RandomState(seed=seed)
        self.sigma_playout = SigmaUniform(rng=self.rng)
        self.eps = eps
        self.delta = sigma


def oos(h, context, tree, pi_i, pi_o, s1, s2, i):
    delta = context.delta
    rng = context.rng
    player = h.player

    if h.is_terminal():
        l = delta*s1 + (1.-delta)*s2
        u = h.utility(i)
        return 1, l, u

    elif player == h.Player.Chance:
        (a, rho1, rho2) = sample_chance(h, context)
        (x, l, u) = oos(h >> a, context, tree,
                        pi_i, rho2 * pi_o,
                        rho1 * s1, rho2 * s2, i)
        return rho2*x, l, u

    infoset = h.infoset()
    (node, out_of_tree) = tree.lookup(infoset)

    if out_of_tree:
        sigma = context.sigma_playout
    else:
        sigma = node.sigma_regret_matching(rng)

    (a, s1_prime, s2_prime) = sample(h, context, infoset, sigma,
                                     s1, s2, i)

    pr_a = sigma.pr(infoset, a)

    if out_of_tree:
        q = delta*s1 + (1.-delta)*s2
        (x, l, u) = playout(h >> a, pr_a*q, sigma)
    else:
        if player == i:
            pi_prime_i = pr_a*pi_i
            pi_prime_o = pi_o
        else:
            pi_prime_i = pi_i
            pi_prime_o = pr_a*pi_o

        (x, l, u) = oos(h >> a, context, tree,
                        pi_prime_i, pi_prime_o,
                        s1_prime, s2_prime, i)

    c = x
    x = pr_a*x

    if player == i:
        w = u*pi_o / l
        for a_prime in infoset.actions:
            if a_prime == a:
                r = (c - x)*w
            else:
                r = - x*w
            node.update_regret(a_prime, r)

    else:
        q = delta*s1 + (1.-delta)*s2
        for a_prime in infoset.actions:
            s = (1./q) * pi_o * sigma.pr(infoset, a_prime)
            node.update_average_strategy(a_prime, s)

    return x, l, u


def solve(h, context, tree, n_iter):
    for i in range(n_iter):
        oos(copy(h), context, tree, 1, 1, 1, 1, h.Player.P1)
        oos(copy(h), context, tree, 1, 1, 1, 1, h.Player.P2)
