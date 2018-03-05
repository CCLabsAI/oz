from oz.game.leduk import LedukPoker

g = LedukPoker()

g.act('bet_or_raise')

g.pretty_print()

g.act('check_or_call')

g.pretty_print()

print("infoset: ", g.infoset())
