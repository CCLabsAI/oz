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

ob = torch.load(os.path.expanduser('~/data/exp/goof6-b-10k/checkpoint-000300.pth'))
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
    json = request.json
    history_str = json['seq']

    h = oz.make_goofspiel2_history(6)

    history_parts = history_str.split('/')
    for part in history_parts:
        if part:
            action_strs = re.split('[<=>? ]', part)
            for action_str in action_strs:
                infoset = h.infoset()
                legal_actions = infoset.actions
                action_index = int(action_str) - 1
                action = next((a for a in legal_actions if a.index == action_index), None)
                if not action:
                    raise "illegal action"
                h.act(action)

    tensor = torch.zeros(encoder.encoding_size())
    infoset = h.infoset()
    encoder.encode(infoset, tensor)
    with torch.no_grad():
        logits = model.forward(tensor)
        probs = F.softmax(logits, dim=0)
    ap = encoder.decode_and_sample(infoset, probs, rng)
    action_index = ap.a.index + 1

    print(logits)
    print(probs)

    ret_json = {
        'action': action_index,
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
    app.run(host='localhost', port=8081, debug=True)
