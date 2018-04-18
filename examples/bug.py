import oz

root = oz.make_leduk_history()
enc = oz.LedukEncoder()
rng = oz.Random()

search_size = 50
bs = oz.BatchSearch(root, enc, search_size)

batch = bs.generate_batch()
print(batch)

enc = None
batch = bs.generate_batch()
print(batch)
