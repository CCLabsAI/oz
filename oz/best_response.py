from copy import copy
from collections import defaultdict


def exploitability(h, sigma):
    depths = infoset_depths(h)
    v1 = gebr(h, h.Player.P1, sigma, depths)
    v2 = gebr(h, h.Player.P2, sigma, depths)
    return v1 + v2


def gebr(h, i, sigma, depths=None):
    if depths is None:
        depths = infoset_depths(h)

    t = defaultdict(float)
    b = defaultdict(float)

    for d in depths:
        gebr_pass2(h, i, d, 0, 1.0, sigma, t, b)

    # final pass should maximize at every depth, so: d = -1
    v = gebr_pass2(h, i, -1, 0, 1.0, sigma, t, b)
    return v


def gebr_pass2(h, i, d, l, pi_o, sigma, t, b):
    if h.is_terminal():
        return h.utility(i)

    player = h.player
    infoset = h.infoset()

    if player is h.Player.Chance:
        v_chance = 0.0
        for a, pr_a in zip(infoset.actions, infoset.probs):
            v_a = gebr_pass2(copy(h) >> a, i, d, l + 1, pi_o * pr_a,
                             sigma, t, b)
            v_chance += pr_a * v_a
        return v_chance

    if player == i and l > d:
        def val(a):
            t_a = t[(infoset, a)]
            b_a = b[(infoset, a)]
            return t_a / b_a if b_a > 0 else 0

        a = max(infoset.actions, key=val)
        return gebr_pass2(copy(h) >> a, i, d, l + 1, pi_o,
                          sigma, t, b)

    v = 0
    for a in infoset.actions:
        pi_prime_o = pi_o
        if player != i:
            pi_prime_o = pi_o*sigma.pr(infoset, a)

        v_prime = gebr_pass2(copy(h) >> a, i, d, l + 1, pi_prime_o,
                             sigma, t, b)

        if player != i:
            v = v + sigma.pr(infoset, a)*v_prime
        elif player == i and l == d:
            t[(infoset, a)] += v_prime*pi_o
            b[(infoset, a)] += pi_o

    return v


def infoset_depths(h):
    depths = set()
    walk_infosets(h, depths, 0)
    depths = list(depths)
    depths.sort(reverse=True)
    return depths


def walk_infosets(h, depths, l):
    if h.is_terminal():
        return

    player = h.player
    infoset = h.infoset()

    if player is not h.Player.Chance:
        depths.add(l)

    for a in infoset.actions:
        walk_infosets(copy(h) >> a, depths, l + 1)
