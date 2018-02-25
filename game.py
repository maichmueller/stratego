import copy

import numpy as np

import helpers
import pieces


class Game:
    def __init__(self, agent0, agent1, game_size="big"):
        self.game_size = game_size
        self.agents = (agent0, agent1)
        if game_size == "small":
            self.types_available = np.array([0, 1] + [2]*3 + [3]*2 + [10] + [11]*2)
            obstacle_positions = [(2, 2)]
            self.game_dim = 5
        elif game_size == "medium":
            obstacle_positions = [(3, 1), (3, 5)]
            self.types_available = np.array([0, 1] + [2]*5 + [3]*3 + [4]*3 + [5]*2 + [6] + [10] + [11]*4)
            self.game_dim = 7
        else:
            obstacle_positions = [(4, 2), (5, 2), (4, 3), (5, 3), (4, 6), (5, 6), (4, 7), (5, 7)]
            self.types_available = np.array([0, 1] + [2]*8 + [3]*5 + [4]*4 + [5]*4 + [6]*4 +
                                            [7]*3 + [8]*2 + [9]*1 + [10] + [11]*6)
            self.game_dim = 10
        self.board = np.empty((self.game_dim, self.game_dim), dtype=object)
        setup0, setup1 = agent0.setup, agent1.setup

        for idx, piece in np.ndenumerate(setup0):
            if piece is not None:
                piece.hidden = False
                self.board[piece.position] = piece
        for idx, piece in np.ndenumerate(setup1):
            if piece is not None:
                piece.hidden = False
                self.board[piece.position] = piece
        for pos in obstacle_positions:
            obs = pieces.Piece(99, 99, pos)
            obs.hidden = False
            self.board[pos] = obs
        agent0.install_board(self.board, reset=True)
        agent1.install_board(self.board, reset=True)

        self.move_count = 1  # agent 1 starts

        self.deadPieces = []
        dead_piecesdict = dict()
        for type_ in set(self.types_available):
            dead_piecesdict[type_] = 0
        self.deadPieces.append(dead_piecesdict)
        self.deadPieces.append(copy.deepcopy(dead_piecesdict))

        self.battleMatrix = helpers.get_battle_matrix()

    def reset(self):
        self.__init__(self.agents[0], self.agents[1], self.game_size)

    def run_game(self):
        game_over = False
        rewards = None
        while not game_over:
            helpers.print_board(self.board)
            rewards = self.run_step()
            if rewards is not None:
                game_over = True
        return rewards

    def run_step(self):
        turn = self.move_count % 2  # player 1 or player 0
        # print("Round: " + str(self.move_count))
        for agent_ in self.agents:
            agent_.move_count = self.move_count

        # if self.move_count > 1000:  # if game lasts longer than 1000 turns => tie
        #     return 0, 0  # each agent gets reward 0
        new_move = self.agents[turn].decide_move()
        # test if agent can't move anymore
        if new_move is None:
            if turn == 1:
                return 2, -2  # agent0 wins
            else:
                return -2, 2  # agent1 wins
        self.do_move(new_move)  # execute agent's choice
        # test if game is over
        if self.goal_test():  # flag discovered
            if turn == 1:
                return -1, 1  # agent1 wins
            elif turn == 0:
                return 1, -1  # agent0 wins
        self.move_count += 1
        return None

    def do_move(self, move):
        """
        :param move: tuple or array consisting of coordinates 'from' at 0 and 'to' at 1
        """
        from_ = move[0]
        to_ = move[1]
        # let agents update their boards too
        for _agent in self.agents:
            _agent.do_move(move, true_gameplay=True)

        if not helpers.is_legal_move(self.board, move):
            return False  # illegal move chosen
        self.board[from_].has_moved = True
        if not self.board[to_] is None:  # Target field is not empty, then has to fight
            fight_outcome = self.fight(self.board[from_], self.board[to_])
            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return False
            elif fight_outcome == 1:
                self.update_board((to_, self.board[from_]))
                self.update_board((from_, None))
            elif fight_outcome == 0:
                self.update_board((to_, None))
                self.update_board((from_, None))
            else:
                self.update_board((from_, None))
                self.update_board((to_, self.board[to_]))
        else:
            self.update_board((to_, self.board[from_]))
            self.update_board((from_, None))

        return True

    def update_board(self, updated_piece):
        """
        :param updated_piece: tuple (piece_board_position, piece_object)
        """
        pos = updated_piece[0]
        piece = updated_piece[1]
        if piece is not None:
            piece.change_position(pos)
        self.board[pos] = piece
        return

    def fight(self, piece_att, piece_def):
        """
        Determine the outcome of a fight between two pieces: 1: win, 0: tie, -1: loss
        add dead pieces to deadFigures
        """
        outcome = self.battleMatrix[piece_att.type, piece_def.type]
        if outcome == 1:
            self.deadPieces[piece_def.team][piece_def.type] += 1
        elif outcome == 0:
            self.deadPieces[piece_def.team][piece_def.type] += 1
            self.deadPieces[piece_att.team][piece_att.type] += 1
        elif outcome == -1:
            self.deadPieces[piece_att.team][piece_att.type] += 1
        return outcome

    def is_legal_move(self, move_to_check):  # TODO: redirect all references to this function to helpers
        return helpers.is_legal_move(self.board, move_to_check)

    def goal_test(self):
        if self.deadPieces[0][0] == 1 or self.deadPieces[1][0] == 1:
            # print('flag captured')
            return True
        else:
            return False

