from oz.game.leduk import LedukPoker
from oz import oos
from oz import best_response

h = LedukPoker()

context = oos.Context()
tree = oos.Tree()

for i in range(5):
    oos.solve(h, context, tree, n_iter=1000)
    sigma = tree.sigma_average_strategy(context.rng)
    ex = best_response.exploitability(h, sigma)
    print("leduk exploitability: {}".format(ex))

