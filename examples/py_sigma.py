import oz

h = oz.make_leduk_history()

tree = oz.Tree()
rng = oz.Random(1)
oos = oz.OOS()

ex = 100

def pr_callback(infoset, action):
    return 1.0 / len(infoset.actions)

py_sigma = oz.make_py_sigma(pr_callback)

ex = oz.exploitability(h, py_sigma)
print(ex)
