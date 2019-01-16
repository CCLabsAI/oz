import oz

h = oz.make_leduc_history()

tree = oz.Tree()
rng = oz.Random(1)
oos = oz.OOS()

ex = 100

while ex > 0.2:
    oos.search(h, 10000, tree, rng)
    sigma = tree.sigma_average()
    ex = oz.exploitability(h, sigma)
    print(ex)
