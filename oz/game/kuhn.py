from enum import Enum
from copy import copy

class Player(Enum):
    Chance = 0
    P1 = 1
    P2 = 2


class Card(Enum):
    NA = None
    Jack = 1
    Queen = 2
    King = 3


class Action(Enum):
    Pass = 'p'
    Bet = 'b'


class ChanceAction(Enum):
    JQ = (Card.Jack, Card.Queen)
    JK = (Card.Jack, Card.King)
    QJ = (Card.Queen, Card.Jack)
    QK = (Card.Queen, Card.King)
    KJ = (Card.King, Card.Jack)
    KQ = (Card.King, Card.Queen)


_P1 = 0
_P2 = 1


class KuhnPoker:
    Player = Player
    POT_MAX = 2

    def __init__(self):
        self.history = []
        self.player = Player.Chance
        self.finished = False
        self.hand = [Card.NA, Card.NA]
        self.folded = [False, False]
        self.pot = [1, 1]

    def __copy__(self):
        cls = self.__class__
        o = cls.__new__(cls)
        o.history = copy(self.history)
        o.player = self.player
        o.finished = self.finished
        o.hand = copy(self.hand)
        o.folded = copy(self.folded)
        o.pot = copy(self.pot)
        return o

    def is_terminal(self):
        return self.finished

    def utility(self):
        assert self.is_terminal()
        if self.folded[_P1]:
            return -self.pot[_P1]
        elif self.folded[_P2]:
            return self.pot[_P2]
        else:
            if self.hand[_P1].value > self.hand[_P2].value:
                return self.pot[_P2]
            else:
                return -self.pot[_P1]

    def infoset(self):
        if self.player == Player.Chance:
            return self.ChanceInfoset()
        else:
            if self.player == Player.P1:
                card = self.hand[_P1]
            elif self.player == Player.P2:
                card = self.hand[_P2]
            return self.PlayerInfoset(self.player, card, copy(self.history))

    def act(self, a):
        self.history.append(a)
        if self.player == Player.Chance:
            self.hand[_P1] = a.value[_P1]
            self.hand[_P2] = a.value[_P2]
            self.player = Player.P1
        elif self.player == Player.P1:
            if a == Action.Pass:
                if self.pot[_P1] < self.pot[_P2]:
                    self.folded[_P1] = True
                    self.finished = True
                else:
                    self.pot[_P1] = self.pot[_P2]
            elif a == Action.Bet:
                if self.pot[_P2] >= self.POT_MAX:
                    self.pot[_P1] = self.pot[_P2]
                    self.finished = True
                else:
                    self.pot[_P1] += 1
            self.player = Player.P2
        elif self.player == Player.P2:
            if a == Action.Pass:
                if self.pot[_P2] < self.pot[_P1]:
                    self.folded[_P2] = True
                    self.finished = True
                else:
                    self.pot[_P2] = self.pot[_P1]
                self.finished = True
            elif a == Action.Bet:
                if self.pot[_P1] >= self.POT_MAX:
                    self.pot[_P2] = self.pot[_P1]
                    self.finished = True
                else:
                    self.pot[_P2] += 1
            self.player = Player.P1

    def __rshift__(self, a):
        self.act(a)
        return self

    class PlayerInfoset:
        actions = list(Action)

        def __init__(self, player, card, history):
            self.player = player
            self.card = card
            self.history = history

        def __str__(self):
            s = self.card.name[0]
            acts = [a.value for a in self.history
                    if a in Action]
            if len(acts) > 0:
                s += '/' + ''.join(acts)
            return s


    class ChanceInfoset:
        actions = list(ChanceAction)
        probs = [1./6]*len(actions)
