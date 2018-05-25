import oz
import oz.nn
import argparse

import torch.nn.functional as F

# history = oz.make_goofspiel2_history(6)
# encoder = oz.make_goofspiel2_encoder(6)

#history = oz.make_leduk_history()
#encoder = oz.make_leduk_encoder()

history = oz.make_liars_dice_history()
encoder = oz.make_liars_dice_encoder()

batch_sigma = oz.SigmaBatch()

batch_sigma.walk_infosets(history)
batch = batch_sigma.generate_batch(encoder)

print(batch);
print(batch.size())

args = argparse.Namespace(nn_arch="mlp", hidden_size=25)

model = oz.nn.model_with_args(args,
    input_size=encoder.encoding_size(),
    output_size=encoder.max_actions())

logits = model.forward(batch)
probs = F.softmax(logits, dim=1)

batch_sigma.store_probs(encoder, probs)

sigma_nn = batch_sigma.make_sigma()
ex1 = oz.gebr(history, oz.P1, sigma_nn)
ex2 = oz.gebr(history, oz.P2, sigma_nn)

print("{:.3f}, {:.3f}, {:.3f}".format(ex1 + ex2, ex1, ex2))
