import random
from copy import copy
from enum import Enum

class Player(Enum):
    Chance = -1
    P1     = 0
    P2     = 1

class Card(Enum):
    Jack  = 1
    Queen = 2
    King  = 3

class Action(Enum):
    Pass = 'p'
    Bet = 'b'
    JJ = (Card.Jack, Card.Jack)
    JQ = (Card.Jack, Card.Queen)
    JK = (Card.Jack, Card.King)
    QJ = (Card.Queen, Card.Jack)
    QQ = (Card.Queen, Card.Queen)
    QK = (Card.Queen, Card.King)
    KJ = (Card.King, Card.Jack)
    KQ = (Card.King, Card.Queen)
    KK = (Card.King, Card.King)
    J = Card.Jack
    Q = Card.Queen
    K = Card.King

player_actions = [Action.Pass, Action.Bet]

preflop_actions = [
    Action.JJ, Action.JQ, Action.JK,
    Action.QJ, Action.QQ, Action.QK,
    Action.KJ, Action.KQ, Action.KK
]

postflop_actions = [Action.J, Action.Q, Action.K]

Deck = [Card.Jack, Card.Queen, Card.King] * 2

def _player_pair(a, b):
    return {Player.P1: a, Player.P2: b}

class LedukPoker:
    Player = Player
    Action = Action
    Card   = Card

    def __init__(self):
        self.deck = copy(Deck)
        self.history = []
        self.showdown = False
        self.folded = _player_pair(False, False)
        self.player = Player.Chance
        self.hand = _player_pair(None, None)
        self.board = None
        self.pot = _player_pair(1, 1)
        self.bet = _player_pair(0, 0)

    def __copy__(self):
        other = type(self)()
        other.deck = copy(self.deck)
        other.history = copy(self.history)
        other.showdown = copy(self.showdown)
        other.folded = copy(self.folded)
        other.player = copy(self.player)
        other.hand = copy(self.hand)
        other.board = copy(self.board)
        other.pot = copy(self.pot)
        other.bet = copy(self.bet)
        return other

    def is_terminal(self):
        return self.showdown or \
               self.folded[Player.P1] or \
               self.folded[Player.P2]

    def legal_actions(self):
        if self.player == Player.Chance:
            if self.hand[Player.P1] is None:
                return preflop_actions
            else:
                return postflop_actions
        else:
            return player_actions

    def act(self, a):
        if self.player == Player.Chance:
            if self.hand[Player.P1] is None:
                self.hand[Player.P1] = a.value[0]
                self.hand[Player.P2] = a.value[1]
                self.player = Player.P1
            else:
                self.board = a.value
                self.showdown = True

        elif self.player == Player.P1:
            if a == Action.Pass and self.bet[Player.P2] > self.bet[Player.P1]:
                self.folded[Player.P1] = True

            if a == Action.Bet:
                self.bet[Player.P1] = self.bet[Player.P2] + 1

            self.player = Player.P2

        elif self.player == Player.P2:
            if a == Action.Pass and self.bet[Player.P1] > self.bet[Player.P2]:
                self.folded[Player.P2] = True

            if a == Action.Bet:
                self.bet[Player.P2] = self.bet[Player.P1] + 1

            self.player = Player.Chance

    def infoset(self):
        if self.player == Player.P1:
            return "P1:"+self.hand[Player.P1].name
        elif self.player == Player.P2:
            return "P2:"+self.hand[Player.P2].name

    def reward(self):
        self.pot[Player.P1] += self.bet[Player.P1]
        self.pot[Player.P2] += self.bet[Player.P2]

        if self.showdown:
            p1_value = self.hand[Player.P1].value
            p2_value = self.hand[Player.P2].value

            if p1_value > p2_value:
                return self.pot[Player.P2]
            elif p1_value < p2_value:
                return -self.pot[Player.P1]
            else:
                return 0

        elif self.folded[Player.P1]:
            return -self.pot[Player.P1]

        elif self.folded[Player.P2]:
            return self.pot[Player.P2]
