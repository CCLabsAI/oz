import random
from collections import defaultdict


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
    # FIXME use real probabilities
    actions = h.infoset().actions
    a = random.choice(actions)
    pr_a = 1./len(actions)
    return a, pr_a, pr_a


def playout(h, s, sigma):
    x = 1.
    while not h.is_terminal():
        infoset = h.infoset()
        a, pr_a = sigma.sample_pr(infoset)
        h = h >> a
        x = pr_a*x
    u = h.utility()
    return x, s*x, u


def _regret_to_pr(regrets):
    total = 0
    probs = {}

    for a, r in regrets.items():
        if r > 0:
            total += r

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
    def pr(self, infoset, a):
        actions = infoset.actions
        return 1./len(actions)

    def sample_pr(self, infoset):
        actions = infoset.actions
        a = random.choice(actions)
        pr_a = 1./len(actions)
        return a, pr_a

    def sample_eps(self, infoset, eps):
        return self.sample_pr(infoset)


class SigmaRegretMatching:
    def __init__(self, regrets):
        self.regrets = regrets
        self._pr = _regret_to_pr(regrets)
        self._action_list = list(self._pr.keys())
        self._prob_list = list(self._pr.values())

    def pr(self, infoset, a):
        return self._pr[a]

    def sample_pr(self, infoset):
        a = random.choices(self._action_list, weights=self._prob_list)[0]
        return a, self._pr[a]

    def sample_eps(self, infoset, eps):
        if random.random() > eps:
            a = random.choices(self._action_list, weights=self._prob_list)[0]
        else:
            a = random.choice(self._action_list)
        pr_a_eps = eps*(1./len(self._action_list)) + (1-eps)*self._pr[a]
        return a, pr_a_eps


class Tree:
    class Node:
        def __init__(self):
            self.regrets = defaultdict(float)
            self.average_strategy = defaultdict(float)

        def sigma_regret_matching(self):
            # TODO Make this less ugly
            if len(self.regrets) > 0:
                return SigmaRegretMatching(self.regrets)
            else:
                return SigmaUniform()

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


class Context:
    def __init__(self):
        self.sigma_playout = SigmaUniform()
        self.eps = 0.2
        self.delta = 0.4


def oss(h, context, tree, pi_i, pi_o, s1, s2, i):
    delta = context.delta
    player = h.player

    if h.is_terminal():
        l = delta*s1 + (1.-delta)*s2
        u = h.utility(i)
        return 1, l, u

    elif player == h.Player.Chance:
        (a, rho1, rho2) = sample_chance(h, context)
        (x, l, u) = oss(h >> a, context, tree,
                        pi_i, rho2*pi_o,
                        rho1*s1, rho2*s2, i)
        return rho2*x, l, u

    infoset = h.infoset()
    (node, out_of_tree) = tree.lookup(infoset)

    if out_of_tree:
        sigma = context.sigma_playout
    else:
        sigma = node.sigma_regret_matching()

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

        (x, l, u) = oss(h >> a, context, tree,
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
