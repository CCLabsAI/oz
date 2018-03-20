from enum import Enum


class Player(Enum):
    Chance = 0
    P1 = 1
    P2 = 2


class Action(Enum):
    Left = 1
    Right = 2
    Heads = -1
    Tails = -2


Action.player_actions = [Action.Left, Action.Right]
Action.chance_actions = [Action.Heads, Action.Tails]
Action.chance_probs = [.5, .5]


class FlipGuess:
    Player = Player
    Action = Action

    def __init__(self):
        self.finished = False
        self.heads = None
        self.p1_choice = None
        self.p2_choice = None
        self.player = Player.Chance

    def is_terminal(self):
        return self.finished

    def utility(self, player=Player.P1):
        if player == Player.P1:
            return self._utility()
        else:
            return -self._utility()

    def _utility(self):
        assert self.is_terminal()
        if self.heads and self.p1_choice == Action.Left:
            return 1
        elif self.p1_choice == self.p2_choice:
            return 3
        else:
            return 0

    def infoset(self):
        if self.player == Player.Chance:
            return self.ChanceInfoset()
        else:
            return self.PlayerInfoset(self.player)

    def act(self, a):
        if self.player == Player.Chance:
            if a == Action.Heads:
                self.heads = True
            elif a == Action.Tails:
                self.heads = False
            else:
                raise ValueError('illegal action: "{}"'.format(a))
            self.player = Player.P1

        elif self.player == Player.P1:
            self.p1_choice = a
            self.player = Player.P2
            if self.heads:
                self.finished = True

        elif self.player == Player.P2:
            self.p2_choice = a
            self.finished = True

        else:
            raise ValueError('illegal action: "{}"'.format(a))

    def __rshift__(self, a):
        self.act(a)
        return self

    class PlayerInfoset:
        actions = Action.player_actions

        def __init__(self, player):
            self.player = player

        def __str__(self):
            return self.player.name

    class ChanceInfoset:
        actions = Action.chance_actions
        probs = Action.chance_probs
