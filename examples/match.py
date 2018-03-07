from oz.game.leduk import LedukPoker
from oz.game.match import Match
from oz.game.agent import BiasedRandomAgent, RandomAgent, CallAgent

players = BiasedRandomAgent(), CallAgent()
m = Match(LedukPoker, players)

returns = []

for i in range(10000):
    returns.append(m.play())

print(sum(returns) / len(returns))
