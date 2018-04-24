from copy import copy

import oz

n_cards = 13

root = oz.make_goofspiel2_history(n_cards)
h = copy(root)

tree = oz.Tree()
rng = oz.Random(1)
oos = oz.OOS()

# force = True
force = False

target = oz.make_goofspiel2_target(oz.P2, n_cards)

oos.search(root, 10000, tree, rng, eps=0.4, delta=0, gamma=0.05)


def print_history(h, actions):
    print("bidding on:", h.game.turn)
    print("score:", (h.game.score(oz.P1), h.game.score(oz.P2)))
    print("bids:", h.game.bids(h.player))
    print("wins:", h.game.wins)


def input_action(actions):
    action_indexes = [a.index for a in actions]

    while True:
        a_str = input("enter card number to play: ")

        try:
            a_idx = int(a_str)
        except ValueError:
            continue

        if a_idx in action_indexes:
            break
        else:
            print("invalid card number")

    a_pos = action_indexes.index(a_idx)
    a = actions[a_pos]
    return a


rng_sigma = oz.Random()

while not h.is_terminal():
    infoset = h.infoset()
    actions = infoset.actions
    action_indexes = [a.index for a in actions]

    print_history(h, actions)
    print("cards in hand:", action_indexes)

    a = input_action(infoset.actions)
    h.act(a)
    target.game.act(a)

    oos.retarget()
    oos.search(root, 5000, tree, rng, target, eps=0.4, delta=0.9, gamma=0.01)


    infoset2 = h.infoset()
    node = tree.lookup(infoset2)
    sigma = tree.sigma_average()

    print("average targeting ratio:", oos.avg_targeting_ratio)
    print("updates at AI node:", node.regret_n)

    assert h.player == oz.P2
    assert target.game.player == oz.P2

    a_probs = [(a.index, sigma.pr(infoset2, a)) for a in infoset2.actions]
    print("AI action probs:", a_probs)

    print("AI regrets:", [(a.index, r)
                           for a, r in node.regrets.items()])

    print("AI avg strategy:", [(a.index, s)
                                for a, s in node.average_strategy.items()])

    ap = sigma.sample_pr(infoset2, rng_sigma)

    if force:
        a = input_action(infoset2.actions)
        h.act(a)
        target.game.act(a)

    else:
        h.act(ap.a)
        target.game.act(ap.a)

    print()

print("game over!")
print("final score:", (h.game.score(oz.P1), h.game.score(oz.P2)))

u = h.utility(oz.P1)
if u > 0:
    print("you win!")
else:
    print("you lose.")
