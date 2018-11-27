import sys
import random
from copy import copy
import argparse
# TODO move me
import oz
import oz.nn
import torch
import torch.nn.functional



class OOSPlayer:
    def __init__(self, history_root, n_iter, eps, delta, gamma, beta):
        self.history_root = copy(history_root)
        self.tree = oz.Tree()
        self.oos = oz.OOS()
        self.n_iter = n_iter
        self.eps = eps
        self.delta = delta
        self.gamma = gamma
        self.beta = beta

    def sample_action(self, infoset, rng):
        sigma = self.tree.sigma_average()
        ap = sigma.sample_pr(infoset, rng)
        return ap.a

    def think(self, infoset, rng):
        # self.oos.retarget()
        self.oos.search(
            self.history_root,
            self.n_iter, self.tree, rng,
            eps=self.eps, delta=self.eps, gamma=self.gamma, beta=self.beta)


class TargetedOOSPlayer(OOSPlayer):
    def __init__(self, history_root, target, **kwargs):
        super().__init__(history_root, **kwargs)
        self.target = target

    def think(self, infoset, rng):
        self.oos.search_targeted(
            self.history_root,
            self.n_iter, self.tree, rng,
            self.target, infoset,
            eps=self.eps, delta=self.eps, gamma=self.gamma)


class NeuralNetPlayer:
  def __init__(self, model, encoder):
    self.model = model
    self.encoder = encoder
    self.encoding_size = encoder.encoding_size()
    
  def sample_action(self, infoset, rng):
      encoder = self.encoder
      infoset_encoded = torch.zeros(self.encoding_size)
      encoder.encode(infoset, infoset_encoded)
      logits = self.model.forward(infoset_encoded.unsqueeze(0))
      probs = logits.exp()
      ap = encoder.decode_and_sample(infoset, probs[0], rng)
      return ap.a
    
  def think(self, infoset, rng):
      pass

class NeuralOOSPlayer:
    def __init__(self, model, encoder, target, history_root,
                 simulation_iter, search_batch_size,
                 eps, delta, gamma, beta):
        self.model = model
        self.encoder = encoder
        self.target = target
        self.history_root = history_root
        self.simulation_iter = simulation_iter
        self.search_batch_size = search_batch_size
        self.eps = eps
        self.delta = delta
        self.gamma = gamma
        self.beta = beta
        self.eta = 1.0
        self.encoding_size = encoder.encoding_size()
        self.batch_search = self.make_batch_search()

    def make_batch_search(self):
        return oz.BatchSearch(batch_size=self.search_batch_size,
                              history=self.history_root,
                              encoder=self.encoder,
                              target=self.target,
                              eps=self.eps, delta=self.delta, gamma=self.gamma,
                              beta=self.beta, eta=self.eta)

    def sample_action(self, infoset, rng):
        sigma = self.batch_search.tree.sigma_average()
        ap = sigma.sample_pr(infoset, rng)
        return ap.a

    def think(self, infoset, rng):
        search = self.batch_search
        search.target(infoset)
        for i in range(self.simulation_iter):
            batch = search.generate_batch()
            
            if len(batch) == 0:
                search.step(rng)
            else:
                with torch.no_grad():
                    logits = self.model.forward(batch)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                search.step(probs, rng)


def load_checkpoint_model(encoder, checkpoint_path):
    ob = torch.load(checkpoint_path)
    encoding_size = encoder.encoding_size()
    max_actions = encoder.max_actions()

    train_args = argparse.Namespace(**ob['args'])
    model = oz.nn.model_with_args(train_args,
                input_size=encoding_size,
                output_size=max_actions)
    model.load_state_dict(ob['model_state'])
    model.eval()
    return model






# It takes two parameters as input : the history string and the path to the checkpoint to be loaded
# The numbers of cards (actions) received by the front end are 1 - 6
# It will return an action in the range 1 - 6 (next_action)

# Example python examples/misc/test_demo.py --history "2<4/1<3/3<5/4<6/" --checkpoint_path 25nov-10k-checkpoint-002500.pth

def main():

    parser = argparse.ArgumentParser(description = "calculate next action ")
    parser.add_argument("--history_string", help = "history of the game", required = True)
    parser.add_argument("--checkpoint_path",
                        help = "nn checkpoint",
                        required = True)
    parser.add_argument("--eps", type=float,
                        help="exploration factor",
                        default=0.4)
    parser.add_argument("--delta", type=float,
                        help="targeting factor",
                        default=0.9)
    parser.add_argument("--gamma", type=float,
                        help="opponent error factor",
                        default=0.01)
    parser.add_argument("--beta", type=float,
                        help="opponent error factor",
                        default=0.99)
    
    parser.add_argument("--search_batch_size", type=int,
                        help="number of concurrent searches when using NN",
                        default=20)

    args = parser.parse_args()
    
    history = None
    target = None
    number_of_cards = 6

    h = oz.make_goofspiel2_history(number_of_cards)
    encoder = oz.make_goofspiel2_encoder(number_of_cards)
    target  = oz.make_goofspiel2_target()

    n_iter = 10000
    search_batch_size = 20

    def make_player( checkpoint_path):
      if checkpoint_path is None:
        print('error: missing checkpoint path', file=sys.stderr)
        exit(1)
      model = load_checkpoint_model(encoder, checkpoint_path)
      return NeuralNetPlayer(model = model, encoder = encoder)
    
    rng = oz.Random()

    # Load player_1 from the checkpoint file

    player_1 = make_player(args.checkpoint_path)


    # Replay all the actions in the history

    infoset = h.infoset()
    original_actions = infoset.actions
    
    for i in range(len(args.history_string)) :
      infoset = h.infoset()
      actions = infoset.actions
      
      if (i % 4 == 0):

        #Previous action of P1
        h.act(original_actions[int(args.history_string[i]) - 1])
        #Previous action of P2
        h.act(original_actions[int(args.history_string[i + 2]) - 1])
        
        #print ('past actions of player_1 : ', args.history_string[i])

    infoset = h.infoset()
    
    player_1.think(infoset, rng)
    
    next_action = player_1.sample_action(infoset, rng)
    next_action_idx = next_action.index + 1

    print('Next action : ', next_action_idx)
    #h.act(next_action)












if __name__ == "__main__":
    main()
