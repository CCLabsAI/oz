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


def assert_loaded(model, load_state):
    new_state = model.state_dict()
    assert new_state.keys() == load_state.keys()
    for k in new_state:
        assert torch.equal(new_state[k].cpu(), load_state[k].cpu()), "{} not equal".format(k)


app = Bottle()

encoder = oz.make_goofspiel2_encoder(6)

ob = torch.load(os.path.expanduser('~/data/goof6-b-10k/checkpoint-000300.pth'))
args = argparse.Namespace(**ob['args'])
state_dict = ob['model_state']

print(args)

model = oz.nn.model_with_args(args,
            input_size=encoder.encoding_size(),
            output_size=encoder.max_actions())

print(model)

model.load_state_dict(state_dict)
assert_loaded(model, state_dict)
model.eval()


rng = oz.Random(1)


@app.route('/api/goofspiel/prediction', method='POST')
def index():
    print(request.json)


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
            if type is not KeyboardInterrupt:
                pdb.post_mortem(tb)
    return wrapper


if __name__ == '__main__':
    app.install(live_debugger)
    app.run(host='localhost', port=8081, debug=True)
