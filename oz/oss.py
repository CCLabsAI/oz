CHANCE = 0


def sample(h, context, infoset, sigma, s1, s2, i):
    # TODO implement targeting
    # a, rho1, rho2 = sigma.sample_targeted(context, infoset, eps)
    # return a, rho1*s1, rho2*s2
    player = h.current_player
    if player == i:
        eps = context.eps
        a, pr_a = sigma.sample_eps(infoset, eps)
        return a, pr_a*s1, pr_a*s2
    else:
        a, pr_a = sigma.sample_prob(infoset)
        return a, pr_a*s1, pr_a*s2


def sample_chance(h, context):
    return ..., ..., ...


def playout(h, s, sigma):
    x = 1.
    while not h.is_terminal():
        infoset = h.infoset()
        a, pr_a = sigma.sample_prob(infoset)
        h = h >> a
        x = pr_a*x
    u = h.utility()
    return x, s*x, u


def regret_matching_policy(r):
    return ...


def uniform_policy(I):
    return ...


def update_regret(tree, info, action, r):
    pass


def update_average_strategy(tree, info, action, s):
    pass


def oss(h, context, tree, pi_i, pi_o, s1, s2, i):
    delta = context.delta
    player = h.current_player

    if h.is_terminal():
        l = delta*s1 + (1.-delta)*s2
        u = h.utility()
        return 1, l, u

    elif player == CHANCE:
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
            s = (1./q) * pi_o * sigma(infoset, a_prime)
            node.update_average_strategy(a_prime, s)

    return x, l, u
