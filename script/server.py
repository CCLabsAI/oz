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
import train_better_hand
import train_opponent_hand

TOP_K = 10

app = Bottle()


def assert_loaded(model, load_state):
    new_state = model.state_dict()
    assert new_state.keys() == load_state.keys()
    for k in new_state:
        assert torch.equal(new_state[k].cpu(), load_state[k].cpu()), "{} not equal".format(k)


def load_model(model, path):
    path = os.path.expanduser(path)
    state_dict = torch.load(path, map_location='cpu')

    model.load_state_dict(state_dict)
    assert_loaded(model, state_dict)
    model.eval()

    print(path)
    print(model)
    return model


def load_oz_model():
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

    print(model)
    return model


utility_model = load_model(train_utility.build_model(),
    "~/src/poker-predict/models/poker_utility_model_demo1.pth")

action_model = load_model(train_actions.build_model(),
    "~/src/poker-predict/models/poker_action_model_demo1.pth")

better_hand_model = load_model(train_better_hand.build_model(),
    "~/src/poker-predict/models/poker_better_hand_model_demo1.pth")

opponent_hand_model = load_model(train_opponent_hand.build_model(),
    "~/src/poker-predict/models/poker_opponent_hand_model_demo1.pth")


rng = oz.Random(1)

OZ_ACTION_NAMES = [None, 'raise', 'call', 'fold']
ACTION_NAMES = ['raise', 'call', 'fold']

def lower_suit(card):
    r, s = card
    return r + s.lower()

RANKS='23456789TJQKA'
SUITS='hcds'

def card_idx_str(card_idx):
    rank = card_idx % log2vector.N_RANKS
    suit = card_idx // log2vector.N_RANKS
    return RANKS[rank] + SUITS[suit]


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

    state = log2vector.parse_state(bytes(hist_str, 'ASCII'))
    print(state)

    state_vector = log2vector.state_to_vector(state)
    state_tensor = torch.from_numpy(state_vector.astype(np.float32))

    with torch.no_grad():
        a_logits = action_model.forward(state_tensor)
        dist = torch.distributions.Categorical(logits=a_logits)
        a = dist.sample().item()
        action_name = ACTION_NAMES[a]

    with torch.no_grad():
        expected_value = utility_model.forward(state_tensor).item()
        expected_value *= train_utility.UTILITY_SCALE

    with torch.no_grad():
        better_hand_logits = better_hand_model.forward(state_tensor)
        better_hand_prob = F.softmax(better_hand_logits)[0].item()
        better_hand_percent = np.round(100 * better_hand_prob)

    with torch.no_grad():
        opponent_hand_logits = opponent_hand_model.forward(state_tensor)
        opponent_hand_probs = F.softmax(opponent_hand_logits)
        oh_sorted, oh_indices = opponent_hand_probs.sort(descending=True)
        top_k = oh_indices[:TOP_K]
        top_k_hands = [log2vector.class_to_cards(i.item())
                        for i in top_k]
        hand_predictions = [[[card_idx_str(i) for i in h]] for h in top_k_hands]

    ret_json = {
        'action': action_name,
        'bluff': better_hand_percent,
        'possibleHands': hand_predictions,
        'moneyProspection': expected_value,
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
    app.install(live_debugger)
    app.run(host='localhost', port=8080, debug=True)
