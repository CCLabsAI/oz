#!/usr/bin/env python

import os
import re
import argparse
import numpy as np

from bottle import Bottle, request

import torch
import torch.nn.functional as F
import oz
import oz.nn

import log2vector
import train_utility
import train_actions

TEST_POSIBLE_HANDS=[[['AC', '2H', '3S', '4D', '5C']], [['AC', 'AH'],['5C', '5D']], [['AC', 'AD'],['5C', '5D', '5S']], [['5C', '5H', '5S']], [['JH', 'JC']]]

app = Bottle()


def assert_loaded(model, load_state):
    new_state = model.state_dict()
    assert new_state.keys() == load_state.keys()
    for k in new_state:
        assert torch.equal(new_state[k].cpu(), load_state[k].cpu()), "{} not equal".format(k)


encoder = oz.make_holdem_encoder()

ob = torch.load(os.path.expanduser('~/data/exp.old/holdem-demo-10k-partial/checkpoint-000005.pth'))
args = argparse.Namespace(**ob['args'])
state_dict = ob['model_state']

print(args)

model = oz.nn.model_with_args(args,
            input_size=encoder.encoding_size(),
            output_size=encoder.max_actions())

model.load_state_dict(state_dict)
assert_loaded(model, state_dict)
model.eval()

utility_model = train_utility.build_model()

path = os.path.expanduser("~/src/poker-predict/models/poker_utility_model_demo1.pth")
state_dict = torch.load(path, map_location='cpu')

utility_model.load_state_dict(state_dict)
assert_loaded(utility_model, state_dict)
utility_model.eval()

print(utility_model)


action_model = train_actions.build_model()

path = os.path.expanduser("~/src/poker-predict/models/poker_action_model_demo1.pth")
state_dict = torch.load(path, map_location='cpu')

action_model.load_state_dict(state_dict)
assert_loaded(action_model, state_dict)
action_model.eval()

print(action_model)


rng = oz.Random(1)

OZ_ACTION_NAMES = [None, 'raise', 'call', 'fold']
ACTION_NAMES0 = ['raise', 'call', 'fold']

def lower_suit(card):
    r, s = card
    return r + s.lower()

@app.route('/api/prediction')
@app.route('/api/prediction', method='POST')
def index():
    print(request.json)

    json = request.json
    seq_no_blind = re.sub("^/rr", "/", json['seq'])
    seq_no_blind = re.sub("^/", "", seq_no_blind)
    ai_hand = "".join([lower_suit(c) for c in json['cards']['AI']])
    player_hand = "".join([lower_suit(c) for c in json['cards']['player']])
    hands = ai_hand + '|' + player_hand
    board = "".join([lower_suit(c) for c in json['cards']['table']])
    hist_str = hands
    if len(board) > 0:
        hist_str += "/" + board
    if len(seq_no_blind) > 0:
        hist_str += ":" + seq_no_blind

    print(hist_str)

    # h = oz.make_holdem_history()
    # g = h.game
    # g.read_history_str(hist_str)
    # tensor = torch.zeros(encoder.encoding_size())
    # infoset = h.infoset()
    # encoder.encode(infoset, tensor)
    # logits = model.forward(tensor)
    # probs = F.softmax(logits, dim=0)
    # ap = encoder.decode_and_sample(infoset, probs, rng)
    # action_name = OZ_ACTION_NAMES[ap.a.index]

    def player_for_log(log):
        fp = log2vector.FIRST_PLAYER[len(log.history)-1]
        p = (fp + len(log.history[-1])) % 2
        return p

    log = log2vector.parse_history(bytes(hist_str, 'ASCII'))
    p = player_for_log(log)

    state = log2vector.State(
                    history=log.history,
                    hole=log.hole[p],
                    board=log.board,
                    utility=None,
                    current_player=p,
                    chosen_action=None)

    print(state)

    state_vector = log2vector.state_to_vector(state)
    state_tensor = torch.from_numpy(state_vector.astype(np.float32))

    a_logits = action_model.forward(state_tensor)
    dist = torch.distributions.Categorical(logits=a_logits)
    a = dist.sample().item()
    action_name = ACTION_NAMES0[a]

    v = utility_model.forward(state_tensor).item()
    v *= train_utility.UTILITY_SCALE

    ret_json = {
        'action': action_name,
        'bluff': 50,
        'possibleHands': TEST_POSIBLE_HANDS,
        'moneyProspection': v,
        'winner': None
    }

    print(ret_json)
    return ret_json


def live_debugger(callback):
    import sys, pdb, traceback
    def wrapper(*args,**kwargs):
        try:
            body = callback(*args,**kwargs)
            return body
        except:
            type, value, tb = sys.exc_info()
            traceback.print_exception(type, value, tb)
            if type is not KeyboardInterrupt:
                pdb.post_mortem(tb)
    return wrapper


if __name__ == '__main__':
    # app.install(live_debugger)
    app.run(host='localhost', port=8080, debug=True)
