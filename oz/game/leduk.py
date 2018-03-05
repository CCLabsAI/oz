from copy import copy
import random
from enum import Enum

class Action(Enum):
    Check = 'check_or_call'
    Call  = Check
    Bet   = 'bet_or_raise'
    Raise = Bet
    Fold  = 'fold'
    Next  = 'next_round'

class Card(Enum):
    Jack = 1
    Queen = 2
    King = 3

class LedukPoker():
    ANTE = 1
    RAISE_PER_ROUND = [2, 4]
    MAX_RAISES_PER_ROUND = [2, 2]
    MAX_ROUNDS = 2
    PAIR_RANK = 10

    CARD_NAMES = {Card.Jack: 'J', Card.Queen: 'Q', Card.King: 'K'}
    DECK = [Card.Jack, Card.Queen, Card.King] * 2

    def __init__(self):
        self.player = 0
        self.hand = [None, None]
        self.board = None
        self.pot = [self.ANTE, self.ANTE]
        self.round = 0
        self.checked = False
        self.raises = 0
        self.history = []
        self.folded = [False, False]
        self.deck = copy(self.DECK)
        random.shuffle(self.deck)
        self._deal_hands()

    def _deal_hands(self):
        self.hand[0] = self.deck.pop()
        self.hand[1] = self.deck.pop()

    def _deal_board_card(self):
        self.board = self.deck.pop()

    def is_terminal(self):
        """Is the game finished"""
        return self.folded[0] or self.folded[1] or self.round == self.MAX_ROUNDS

    def player(self):
        """who is the current player"""
        if self.player == 0:
            return 'P1'
        else:
            return 'P2'

    def legal_actions(self):
        """a list of legal actions for the current player"""
        return ['pass', 'bet']

    def act(self, a):
        """
        perform an action
        """
        self.history.append(a)

        if a == Action.Fold:
            self.folded[self.player] = True

        elif a == Action.Call:
            self.pot[self.player] = self.pot[self._other_player()]

            if self.checked or self.raises > 0:
                self._start_next_round()
            else:
                self.checked = True

        elif a == Action.Bet:
            if self.raises >= self.MAX_RAISES_PER_ROUND[self.round]:
                raise ValueError("maximum raises reached")

            other_pot = self.pot[self._other_player()]
            self.pot[self.player] = other_pot + self.RAISE_PER_ROUND[self.round]
            self.raises += 1

        else:
            raise ValueError("'{}' is not a valid action".format(a))

        self.player = self._other_player()


    def infoset(self):
        """return the current infoset wrt the current player"""
        card_info = self.CARD_NAMES[self.hand[self.player]]
        if self.board is not None:
            card_info += self.CARD_NAMES[self.board]
        return "{}:{}".format(card_info, self.history_short())

    def history_short(self):
        s = "/"
        for a in self.history:
            if a == Action.Bet:
                s += 'r'
            elif a == Action.Call:
                s += 'c'
            elif a == Action.Fold:
                s += 'f'
            elif a == Action.Next:
                s += '/'
        return s

    def pretty_print(self):
        print("pot: {}".format(self.pot))
        print("hand: [{}, {}]".format(
            self.CARD_NAMES[self.hand[0]],
            self.CARD_NAMES[self.hand[1]]
        ))
        print("history: {}".format(self.history_short()))

    def reward(self):
        """return the reward at the current node"""
        if self.folded[0]:
            return -self.pot[0]
        elif self.folded[1]:
            return self.pot[1]
        else:
            winner = self._best_hand()
            if winner is None:
                return 0
            elif winner == 0:
                return self.pot[1]
            else:
                return -self.pot[0]

    def _other_player(self):
        if self.player == 0:
            return 1
        else:
            return 0

    def _start_next_round(self):
        self.history.append(Action.Next)
        self.raises = 0
        self.checked = False
        if self.round == 0:
            self._deal_board_card()
        self.round += 1

    def _rank_hand(self, card, board):
        rank = card.value
        if card == board:
            rank += self.PAIR_RANK
        return rank

    def _best_hand(self):
        rank1 = self._rank_hand(self.hand[0], self.board)
        rank2 = self._rank_hand(self.hand[1], self.board)
        if rank1 > rank2:
            return 0
        elif rank2 > rank1:
            return 1
        else:
            return None
