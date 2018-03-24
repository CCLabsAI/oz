from copy import copy
from enum import Enum


class Player(Enum):
    Chance = 0
    P1 = 1
    P2 = 2


class Action(Enum):
    Call = 'c'
    Raise = 'r'
    Fold = 'f'
    NextRound = '/'


Action.Bet = Action.Raise
Action.Check = Action.Call

Action.player_actions = [Action.Call, Action.Raise, Action.Fold]
Action.fold_call = [Action.Fold, Action.Call]
Action.fold_call_raise = [Action.Fold, Action.Call, Action.Raise]


class Card(Enum):
    NA = None
    Jack = 1
    Queen = 2
    King = 3

    @property
    def short_name(self):
        return self.name[0]


class ChanceAction(Enum):
    J1 = (Card.Jack, Player.P1)
    Q1 = (Card.Queen, Player.P1)
    K1 = (Card.King, Player.P1)
    J2 = (Card.Jack, Player.P2)
    Q2 = (Card.Queen, Player.P2)
    K2 = (Card.King, Player.P2)
    J = (Card.Jack, None)
    Q = (Card.Queen, None)
    K = (Card.King, None)


ChanceAction.P1_deal = [ChanceAction.J1, ChanceAction.Q1, ChanceAction.K1]
ChanceAction.P2_deal = [ChanceAction.J2, ChanceAction.Q2, ChanceAction.K2]
ChanceAction.board_deal = [ChanceAction.J, ChanceAction.Q, ChanceAction.K]


_P1 = 0
_P2 = 1


class LedukPoker:
    Player = Player
    Action = Action
    ChanceAction = ChanceAction

    ANTE = 1
    RAISE_PER_ROUND = [2, 4]
    MAX_RAISES = 2
    N_ROUNDS = 2
    PAIR_RANK = 10

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
        self.player = Player.Chance
        self.hand = [Card.NA, Card.NA]
        self.board = Card.NA
        self.pot = [self.ANTE, self.ANTE]
        self.round = 0
        self.checked = False
        self.raises = 0
        self.history = []
        self.folded = [False, False]

    def __copy__(self):
        cls = self.__class__
        o = cls.__new__(cls)
        o.player = self.player
        o.hand = copy(self.hand)
        o.board = self.board
        o.pot = copy(self.pot)
        o.round = self.round
        o.checked = self.checked
        o.raises = self.raises
        o.history = copy(self.history)
        o.folded = copy(self.folded)
        return o

    def is_terminal(self):
        """Is the game finished?"""
        return \
            self.folded[_P1] or \
            self.folded[_P2] or \
            self.round == self.N_ROUNDS

    def act(self, a):
        """perform an action"""
        if self.player is Player.Chance:
            card, deal_player = a.value
            if deal_player == Player.P1:
                self.hand[_P1] = card
            elif deal_player == Player.P2:
                self.hand[_P2] = card
                self.player = Player.P1
            elif deal_player is None:
                self.board = card
                self.player = Player.P1
            else:
                assert False

        else:
            self.history.append(a)

            if a == Action.Fold:
                self.folded[self._player_idx] = True

            elif a == Action.Call:
                self.pot[self._idx(self.player)] = self.pot[self._other_player_idx]

                if self.checked or self.raises > 0:
                    self._start_next_round()
                else:
                    self.checked = True
                    self.player = self._other_player

            elif a == Action.Raise:
                if self.raises >= self.MAX_RAISES:
                    raise ValueError('maximum raises reached')

                other_pot = self.pot[self._other_player_idx]
                self.pot[self._player_idx] = other_pot + self.RAISE_PER_ROUND[self.round]
                self.raises += 1
                self.player = self._other_player

            else:
                raise ValueError('"{}" is not a valid action'.format(a))

    def __rshift__(self, a):
        self.act(a)
        return self

    @staticmethod
    def _idx(player):
        if player == Player.P1:
            return 0
        else:
            return 1

    def infoset(self):
        """return the current infoset for the current player"""
        if self.player is Player.Chance:
            if self.hand[_P1] == Card.NA:
                return self.ChanceInfoset(deal_player=Player.P1)
            elif self.hand[_P2] == Card.NA:
                return self.ChanceInfoset(deal_player=Player.P2)
            elif self.board == Card.NA:
                return self.ChanceInfoset(deal_player=None)
            else:
                assert False
        else:
            card = self.hand[self._idx(self.player)]
            return self.PlayerInfoset(self.player, card, self.board, self.history, self.pot, self.raises)

    def _utility(self):
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

    def utility(self, target_player=Player.P1):
        if target_player == Player.P1:
            return self._utility()
        else:
            return -self._utility()

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
        print('hand: {}'.format(self.hand[self._player_idx].short_name))
        if self.board is not None:
            print('board card: {}'.format(self.board.short_name))
        print('pot: ${}'.format(self.pot[0] + self.pot[1]))
        if self.checked or self.raises > 0:
            print('to call: ${}'.format(self.pot[self._other_player_idx] - self.pot[self._player_idx]))
        print('history: {}'.format(self.history_short()))

    class PlayerInfoset:
        actions = Action.player_actions

        def __init__(self, player, card, board, history, pot, raises):
            self.player = player
            self.card = card
            self.board = board
            self.history = tuple(history)
            self.pot = pot
            self.raises = raises

        def __str__(self):
            s = self.card.short_name
            s += self.board.short_name
            acts = [a.value for a in self.history
                    if a in Action]
            if len(acts) > 0:
                s += '/' + ''.join(acts)
            return s

        @property
        def actions(self):
            if self.raises < LedukPoker.MAX_RAISES:
                return Action.fold_call_raise
            else:
                return Action.fold_call

        def __eq__(self, other):
            return self.player == other.player and \
                    self.card == other.card and \
                    self.board == other.board and \
                    self.history == other.history

        def __hash__(self):
            return hash((self.player, self.card, self.board, self.history))

    class ChanceInfoset:
        def __init__(self, deal_player=None, hand=(Card.NA, Card.NA)):
            self.deal_player = deal_player
            self.hand = hand

        @property
        def actions(self):
            if self.deal_player == Player.P1:
                return ChanceAction.P1_deal
            elif self.deal_player == Player.P2:
                return ChanceAction.P2_deal
            else:
                return ChanceAction.board_deal

        @property
        def probs(self):
            if self.deal_player == Player.P1:
                n = len(ChanceAction.P1_deal)
                return [1./n]*n
            elif self.deal_player == Player.P2:
                n = len(ChanceAction.P2_deal)
                return [1./n]*n
            else:
                n = len(ChanceAction.board_deal)
                return [1./n]*n

        def __eq__(self, other):
            return self.deal_player == other.deal_player

        def __hash__(self):
            return hash(self.deal_player)

    @property
    def _other_player(self):
        if self.player == Player.P1:
            return Player.P2
        else:
            return Player.P1

    @property
    def _player_idx(self):
        if self.player == Player.P1:
            return 0
        else:
            return 1

    @property
    def _other_player_idx(self):
        if self.player == Player.P1:
            return 1
        else:
            return 0

    def _start_next_round(self):
        self.history.append(Action.NextRound)
        self.raises = 0
        self.checked = False
        self.round += 1
        self.player = Player.Chance

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
