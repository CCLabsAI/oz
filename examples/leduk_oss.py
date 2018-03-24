from oz.game.leduk import LedukPoker
from oz import oss
from oz import best_response

h = LedukPoker()

context = oss.Context()
tree = oss.Tree()

for i in range(1000):
    oss.solve(h, context, tree, n_iter=1000)
    sigma = tree.sigma_average_strategy(context.rng)
    ex = best_response.exploitability(h, sigma)
    print("leduk exploitability: {}".format(ex))

