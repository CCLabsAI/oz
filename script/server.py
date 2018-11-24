#!/usr/bin/env python

import os
import re
import argparse

from bottle import Bottle, request

import torch
import torch.nn.functional as F
import oz
import oz.nn


TEST_POSIBLE_HANDS=[[['AC', '2H', '3S', '4D', '5C']], [['AC', 'AH'],['5C', '5D']], [['AC', 'AD'],['5C', '5D', '5S']], [['5C', '5H', '5S']], [['JH', 'JC']]]

app = Bottle()

encoder = oz.make_holdem_encoder()

ob = torch.load(os.path.expanduser('~/data/misc/checkpoint-000005.pth'))
args = argparse.Namespace(**ob['args'])

print(args)

model = oz.nn.model_with_args(args,
            input_size=encoder.encoding_size(),
            output_size=encoder.max_actions())

print(model)

model.load_state_dict(ob['model_state'])

rng = oz.Random(1)

ACTION_NAMES = [None, 'raise', 'call', 'fold']

def lower_suit(card):
    r, s = card
    return r + s.lower()

@app.route('/api/prediction')
@app.route('/api/prediction', method='POST')
def index():
    print(request.json)

    json = request.json
    seq_no_blind = re.sub("^/bb", "/", json['seq'])
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

    h = oz.make_holdem_history()
    g = h.game
    g.read_history_str(hist_str)
    tensor = torch.zeros(encoder.encoding_size())
    infoset = h.infoset()
    encoder.encode(infoset, tensor)
    logits = model.forward(tensor)
    probs = F.softmax(logits, dim=0)
    ap = encoder.decode_and_sample(infoset, probs, rng)
    action_name = ACTION_NAMES[ap.a.index]

    return {
        'action': action_name,
        'bluff': 50,
        'possibleHands': TEST_POSIBLE_HANDS,
        'moneyProspection': 100,
        'winner': None
    }

@app.route('/api/winner', method='POSTq')
def index():
    print(request.json)
    return {
        'winner': 'AI',
        'reason': 'you smell',
        'AIHand': 'super duper pooper',
        'playerHand': 'smelly nutz'
    }

app.run(host='localhost', port=8080, debug=True)
