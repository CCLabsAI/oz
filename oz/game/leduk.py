from copy import copy

import random
from enum import Enum


class Action(Enum):
    Call = 'c'
    Raise = 'r'
    Fold = 'f'
    NextRound = '/'


Action.Bet = Action.Raise
Action.Check = Action.Call

Action.player_actions = [Action.Call, Action.Raise, Action.Fold]


class Card(Enum):
    NA = None
    Jack = 1
    Queen = 2
    King = 3

    @property
    def short_name(self):
        return self.name[0]


class LedukPoker:
    ANTE = 1
    RAISE_PER_ROUND = [2, 4]
    MAX_RAISES = 2
    N_ROUNDS = 2
    PAIR_RANK = 10

    FOLD_CALL = [Action.Fold, Action.Call]
    FOLD_CALL_RAISE = [Action.Fold, Action.Call, Action.Raise]

    DECK = [Card.Jack, Card.Queen, Card.King] * 2

    __slots__ = (
        'player',
        'hand',
        'board',
        'pot',
        'round',
        'checked',
        'raises',
        'history',
        'folded',
        'deck'
    )

    def __init__(self):
        self.player = 0
        self.hand = [Card.NA, Card.NA]
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
        """Is the game finished?"""
        return \
            self.folded[0] or \
            self.folded[1] or \
            self.round == self.N_ROUNDS

    def legal_actions(self):
        """a list of legal actions for the current player"""
        if self.raises < self.MAX_RAISES:
            return self.FOLD_CALL_RAISE
        else:
            return self.FOLD_CALL

    def act(self, a):
        """perform an action"""
        self.history.append(a)

        if a == Action.Fold:
            self.folded[self.player] = True

        elif a == Action.Call:
            self.pot[self.player] = self.pot[self._other_player]

            if self.checked or self.raises > 0:
                self._start_next_round()
            else:
                self.checked = True

        elif a == Action.Raise:
            if self.raises >= self.MAX_RAISES:
                raise ValueError('maximum raises reached')

            other_pot = self.pot[self._other_player]
            self.pot[self.player] = other_pot + self.RAISE_PER_ROUND[self.round]
            self.raises += 1

        else:
            raise ValueError('"{}" is not a valid action'.format(a))

        self.player = self._other_player

    def infoset(self):
        """return the current infoset for the current player"""
        card_info = self.hand[self.player].short_name
        if self.board is not None:
            card_info += self.board.short_name
        return "{}:{}".format(card_info, self.history_short())

    def reward(self):
        """return the reward at the current node"""
        if not self.is_terminal():
            raise RuntimeError('tried to get reward from unfinished game')

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

    def history_short(self):
        return "".join([a.value for a in self.history])

    def pretty_print_state(self):
        print('pot: {}'.format(self.pot))
        print('hands: [{}, {}]'.format(
            self.hand[0].short_name,
            self.hand[1].short_name
        ))
        if self.board is not None:
            print('board card: {}'.format(self.board.short_name))
        print('history: {}'.format(self.history_short()))

    def pretty_print_infoset(self):
        print('hand: {}'.format(self.hand[self.player].short_name))
        if self.board is not None:
            print('board card: {}'.format(self.board.short_name))
        print('pot: ${}'.format(self.pot[0] + self.pot[1]))
        if self.checked or self.raises > 0:
            print('to call: ${}'.format(self.pot[self._other_player] - self.pot[self.player]))
        print('history: {}'.format(self.history_short()))

    @property
    def _other_player(self):
        if self.player == 0:
            return 1
        else:
            return 0

    def _start_next_round(self):
        self.history.append(Action.NextRound)
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
