class Match():
    def __init__(self, Game, players):
        self.Game = Game
        self.player = 0
        self.players = players

    def play(self):
        g = self.Game()
        while not g.is_terminal():
            a = self.players[self.player].choose_action(g)
            g.act(a)
            self.player = self._other_player
        self.g = g
        return g.reward()

    @property
    def _other_player(self):
        if self.player == 0:
            return 1
        else:
            return 0
