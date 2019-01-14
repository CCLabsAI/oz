from oz.game.leduc import LeducPoker, Action

g = LeducPoker()

def input_action(legal_actions):
    while True:
        a_str = input('> ')
        a = next((a for a in legal_actions if a.value == a_str), None)
        if a is not None:
            return a
        else:
            print('invalid action!')

while True:
    print('infoset: "{}"'.format(g.infoset()))
    g.pretty_print_infoset()

    a = input_action(g.legal_actions())
    g.act(a)

    if g.is_terminal():
        break


r = g.reward()

if r > 0:
    print("player 1 wins ${}".format(r))
elif r < 0:
    print("player 2 wins ${}".format(-r))
else:
    print("showdown is a draw, split pot!")
