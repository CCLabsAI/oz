from oz.game.leduk import LedukPoker, Action

g = LedukPoker()

g.act(Action.Raise)

g.pretty_print_state()

g.act(Action.Call)

g.pretty_print_state()

print("infoset: ", g.infoset())
