from oz.game.leduk import LedukPoker
from oz import oos
from oz import best_response
import cProfile

h = LedukPoker()

context = oos.Context()
tree = oos.Tree()

cProfile.run("oss.solve(h, context, tree, n_iter=5000)", sort='tottime')
