#!/usr/bin/env python

from bottle import Bottle, request

import torch
import oz

TEST_POSIBLE_HANDS=[[['AC', '2H', '3S', '4D', '5C']], [['AC', 'AH'],['5C', '5D']], [['AC', 'AD'],['5C', '5D', '5S']], [['5C', '5H', '5S']], [['JH', 'JC']]]

app = Bottle()

@app.route('/api/prediction')
@app.route('/api/prediction', method='POST')
def index():
    print(request.json)
    return {
        'action': 'call',
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
