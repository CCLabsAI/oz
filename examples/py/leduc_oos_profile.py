from oz.game.leduc import LeducPoker
from oz import oos
from oz import best_response
import cProfile

h = LeducPoker()

context = oos.Context()
tree = oos.Tree()

cProfile.run("oos.solve(h, context, tree, n_iter=5000)", sort='tottime')
