import oz

# h = oz.make_kuhn_history()
h = oz.make_leduk_history()

tree = oz.Tree()
rng = oz.Random(1)
oos = oz.OOS()

for i in range(100):
    oos.search(h, 10000, tree, rng)
    sigma = tree.sigma_average()
    ex = oz.exploitability(h, sigma)
    print(ex)
