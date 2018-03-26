from oz.game.leduk import LedukPoker
from oz import oss
from oz import best_response
import cProfile

h = LedukPoker()

context = oss.Context()
tree = oss.Tree()

cProfile.run("oss.solve(h, context, tree, n_iter=5000)", sort='tottime')
