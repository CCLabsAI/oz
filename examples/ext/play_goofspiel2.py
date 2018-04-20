from copy import copy

import oz

root = oz.make_goofspiel2_history(6)
h = copy(root)

tree = oz.Tree()
rng = oz.Random(1)
oos = oz.OOS()

target = oz.make_goofspiel2_target(oz.P2, 6)

oos.search(root, 10000, tree, rng, eps=0.4, delta=0, gamma=0.05)

sigma = tree.sigma_average()

def print_history(h, actions):
    print("bidding on:", h.game.turn)
    print("score:", (h.game.score(oz.P1), h.game.score(oz.P2)))
    print("bids:", h.game.bids(h.player))
    print("wins:", h.game.wins)

rng_sigma = oz.Random()

while not h.is_terminal():
    infoset = h.infoset()
    actions = infoset.actions
    action_indexes = [a.index for a in actions]

    print_history(h, actions)
    print("cards in hand:", action_indexes)

    while True:
        a_str = input("enter card number to play: ")
        a_idx = int(a_str)

        if a_idx in action_indexes:
            break
        else:
            print("invalid card number")

    a_pos = action_indexes.index(a_idx)
    a = actions[a_pos]
    h.act(a)
    target.game.act(a)

    oos.search(root, 5000, tree, rng, target, eps=0.2, delta=1.0, gamma=0.01)
    # oos.search(root, 5000, tree, rng, eps=0.2, delta=0.6, gamma=0.01)

    infoset2 = h.infoset()
    node = tree.lookup(infoset2)
    print("avgerage targeting ratio:", oos.avg_targeting_ratio)
    print("updates at AI node:", node.regret_n)

    assert h.player == oz.P2
    assert target.game.player == oz.P2

    ap = sigma.sample_pr(infoset2, rng_sigma)
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
