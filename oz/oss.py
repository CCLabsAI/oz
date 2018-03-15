CHANCE = 0


def sample(h, I, i, eps):
    return ..., ..., ...


def playout(h, s):
    return ..., ..., ...


def regret_matching(r):
    return ...


def sample_chance(h):
    return ..., ..., ...


def update_regret(tree, info, action, z):
    pass


def update_average_strategy(tree, info, action, z):
    pass


def oss(tree, ctx, h, pi_i, pi_x, s1, s2, i):
    delta = ctx.delta
    P = h.player()
    Px = ...
    if h.is_terminal():
        return 1, delta*s1 + (1 - delta)*s2, h.utility()
    elif P == CHANCE:
        (a, rho1, rho2) = sample_chance(h)
        return oss(tree, ctx, h+a, pi_i, rho2*pi_x, rho1*s1, rho2*s2, i)

    I = h.infoset()
    (a, s1_prime, s2_prime) = sample(h, I, i, ctx.eps)
    if I not in tree:
        tree.append(I)
        sigma = ...
        (x, l, u) = playout(h + a, delta*s1 + (1 - delta)*s2)
    else:
        sigma = regret_matching(tree.regret(I))
        pi_prime_P = sigma(I, a) * pi(P)
        pi_prime_Px = pi(Px)
        (x, l, u) = oss(tree, ctx, h+a, pi_prime_P, pi_prime_Px,
                        s1_prime, s2_prime, i)

    c = x
    x = x * sigma(I, a)
    for a_prime in I.actions:
        if h.player() == i:
            W = u*pi_x / l
            if a_prime == a:
                update_regret(tree, I, a_prime, (c - x)*W)
            else:
                update_regret(tree, I, a_prime, - x*W)
        else:
            z = (1./(delta*s1 + (1-delta)*s2))*pi_x*sigma(I, a_prime)
            update_average_strategy(tree, I, a_prime, z)

    return x, l, u
