import copy

import numpy as np

from cythonize import helpers
import pieces
from collections import Counter, defaultdict


class Game:

    bm = helpers.get_bm()

    def __init__(self, agent0, agent1, board_size="big", fixed_setups=False, *args):
        self.board_size = board_size
        self.agents = (agent0, agent1)
        self.fixed_setups = fixed_setups

        helpers.GameDef.set_board_size(board_size)
        self.obstacle_positions, self.types_available, self.game_dim = helpers.GameDef.get_game_specs()

        if not fixed_setups:
            agent0.setup = self.__draw_random_setup(self.types_available, 0, self.game_dim)
            agent1.setup = self.__draw_random_setup(self.types_available, 1, self.game_dim)

        if agent0.setup is not None and agent1.setup is not None:
            board = self._build_board_from_setups(agent0.setup, agent1.setup)
        else:
            raise ValueError('Missing board information.')

        self.state = GameState(board)
        agent0.install_board(self.state.board, reset=True)
        agent1.install_board(self.state.board, reset=True)

        self.game_replay = GameReplay(self.state.board)

        self.move_count = 1  # agent 1 starts

    def _build_board_from_setups(self, setup0, setup1):
        board = np.empty((self.game_dim, self.game_dim), dtype=object)

        for setup in (setup0, setup1):
            for idx, piece in np.ndenumerate(setup):
                if piece is not None:
                    board[piece.position] = piece

        for pos in self.obstacle_positions:
            obs = pieces.Piece(99, 99, pos)
            obs.hidden = False
            board[pos] = obs

        return board

    def reset(self):
        self.__init__(self.agents[0], self.agents[1], self.board_size)

    def run_game(self, show=False):
        game_over = False
        rewards = None
        if show:
            print_board = helpers.print_board
        else:
            def print_board(*args): pass

        while not game_over:
            print_board(self.state.board)
            rewards = self.run_step()
            if rewards:
                game_over = True
        print_board(self.state.board)
        return rewards

    def run_step(self):
        turn = self.move_count % 2  # player 1 or player 0

        new_move = self.agents[turn].decide_move()

        # test if agent can't move anymore
        if new_move is None:
            if turn == 1:
                return 2  # agent0 wins
            else:
                return -2  # agent1 wins

        # let agents update their boards
        for _agent in self.agents:
            _agent.do_move(new_move, true_gameplay=True)
        self.state.do_move(new_move)  # execute agent's choice

        # test if game is over
        terminal = self.state.is_terminal(flag_only=True, move_count=self.move_count)
        if terminal:  # flag discovered, or draw
            return terminal

        self.move_count += 1
        for agent_ in self.agents:
            agent_.move_count = self.move_count
        return 0

    @staticmethod
    def __draw_random_setup(types_available, team, game_dim):
        """
        Draw a random setup from the set of types types_available after placing the flag
        somewhere in the last row of the board of the side of 'team', or behind the obstacle.
        :param types_available: list of types to draw from, integers
        :param team: boolean, 1 or 0 depending on the team
        :param game_dim: int, the board dimension
        :return: the setup, in numpy array form
        """
        nr_pieces = len(types_available)-1
        types_available = [type_ for type_ in types_available if not type_ == 0]
        if game_dim == 5:
            row_offset = 2
        elif game_dim == 7:
            row_offset = 3
        else:
            row_offset = 4
        setup_agent = np.empty((row_offset, game_dim), dtype=object)
        if team == 0:
            flag_positions = [(game_dim-1, j) for j in range(game_dim)]
            flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
            flag_pos = game_dim-1 - flag_positions[flag_choice][0], game_dim-1 - flag_positions[flag_choice][1]
            setup_agent[flag_pos] = pieces.Piece(0, 0, flag_positions[flag_choice])

            types_draw = np.random.choice(types_available, nr_pieces, replace=False)
            positions_agent_0 = [(i, j) for i in range(game_dim-row_offset, game_dim) for j in range(game_dim)]
            positions_agent_0.remove(flag_positions[flag_choice])

            for idx in range(nr_pieces):
                pos = positions_agent_0[idx]
                setup_agent[(game_dim-1 - pos[0], game_dim-1 - pos[1])] = pieces.Piece(types_draw[idx], 0, pos)
        elif team == 1:
            flag_positions = [(0, j) for j in range(game_dim)]
            flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
            setup_agent[flag_positions[flag_choice]] = pieces.Piece(0, 1, flag_positions[flag_choice])

            types_draw = np.random.choice(types_available, nr_pieces, replace=False)
            positions_agent_1 = [(i, j) for i in range(row_offset) for j in range(game_dim)]
            positions_agent_1.remove(flag_positions[flag_choice])

            for idx in range(nr_pieces):
                pos = positions_agent_1[idx]
                setup_agent[pos] = pieces.Piece(types_draw[idx], 1, pos)
        return setup_agent


class GameState:
    def __init__(self, board=None, dead_pieces=None, move_count=None):

        self.obstacle_positions = None
        if dead_pieces is not None:
            self.dead_pieces = dead_pieces
        else:
            self.dead_pieces = (dict(), dict())
        self.board = board
        self.game_dim = board.shape[0]
        self.obstacle_positions = helpers.GameDef.get_game_specs()[0]

        self.move_count = move_count

        self.terminal = 0
        self.check_terminal()
        self.terminal_checked = True

        self.dead_pieces = dict()
        pieces0, pieces1 = defaultdict(int), defaultdict(int)
        for piece in board.flatten():
            if piece is not None:
                if piece.team:
                    pieces1[piece.type] += 1
                else:
                    pieces0[piece.type] += 1

        for pcs, team in zip((pieces0, pieces0), (0, 1)):
            dead_pieces_dict = dict()
            for type_, freq in Counter(helpers.GameDef.get_game_specs()[1]).items():
                dead_pieces_dict[type_] = freq - pcs[type_]
            self.dead_pieces[team] = dead_pieces_dict

    def update_board(self, pos, piece):
        """
        :param pos: tuple piece board position
        :param piece: the new piece at the position
        """
        if piece is not None:
            piece.change_position(pos)
        self.board[pos] = piece
        self.terminal_checked = False
        return

    def check_terminal(self, flag_only=False, turn=None):
        if not any(self.dead_pieces):
            flags = sum([piece.team + 1 for piece in self.board.flatten() if piece is not None and piece.type == 0])
            if flags != 3:  # == 3 only if both flag 0 and flag 1 are present
                if flags == 1:  # agent 1 flag has been captured
                    self.terminal = 1  # agent 0 wins by flag
                else:
                    self.terminal = -1  # agent 1 wins by flag

        else:
            self.terminal =  1 * (self.dead_pieces[0][0] == 1)
            self.terminal = -1 * (self.dead_pieces[1][0] == 1)

        if not flag_only:
            if turn is None:
                turn = 0
            if not helpers.get_poss_moves(self.board, turn):
                self.terminal = -2  # agent 1 wins by moves
            elif not helpers.get_poss_moves(self.board, (turn + 1) % 2):
                self.terminal = 2  # agent 0 wins by moves

        if self.move_count is not None and self.move_count > 500:
            self.terminal = 404

        self.terminal_checked = True

    def do_move(self, move):
        """
        :param move: tuple or array consisting of coordinates 'from' at 0 and 'to' at 1
        """
        from_ = move[0]
        to_ = move[1]

        fight_outcome = None

        board = self.board

        # if not helpers.is_legal_move(self.board, move):
        #    return False  # illegal move chosen

        board[from_].has_moved = True

        if not board[to_] is None:  # Target field is not empty, then has to fight
            board[from_].hidden = board[to_].hidden = False
            fight_outcome = self.fight(board[from_], board[to_])
            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return False
            elif fight_outcome == 1:
                self.update_board(to_, board[from_])
                self.update_board(from_, None)
            elif fight_outcome == 0:
                self.update_board(to_, None)
                self.update_board(from_, None)
            else:
                self.update_board(from_, None)
                self.update_board(to_, board[to_])
        else:
            self.update_board(to_, board[from_])
            self.update_board(from_, None)
        # self.game_replay.add_move(move, (board[from_], board[to_]), self.move_count % 2, self.move_count)
        return fight_outcome

    def fight(self, piece_att, piece_def):
        """
        Determine the outcome of a fight between two pieces:
        1: win, 0: tie, -1: loss
        add dead pieces to deadFigures
        """
        outcome = Game.bm[piece_att.type, piece_def.type]
        if outcome == 1:
            self.dead_pieces[piece_def.team][piece_def.type] += 1
        elif outcome == 0:
            self.dead_pieces[piece_def.team][piece_def.type] += 1
            self.dead_pieces[piece_att.team][piece_att.type] += 1
        elif outcome == -1:
            self.dead_pieces[piece_att.team][piece_att.type] += 1
        return outcome

    def is_terminal(self, **kwargs):
        if not self.terminal_checked:
            self.check_terminal(*kwargs)
        return self.terminal


class GameReplay:
    def __init__(self, board):
        self.initialBoard = copy.deepcopy(board)
        self.curr_board = copy.deepcopy(board)
        self.pieces_team_0 = []
        self.pieces_team_1 = []
        for pos, piece in np.ndenumerate(self.initialBoard):
            if piece is not None:
                if piece.team == 0:
                    self.pieces_team_0.append(piece)
                else:
                    self.pieces_team_1.append(piece)
        self.moves_and_pieces_in_round = dict()
        self.team_of_round = dict()

    def add_move(self, move, pieces, team, round):
        self.moves_and_pieces_in_round[round] = (move, pieces[0], pieces[1])
        self.team_of_round[round] = team
        self.curr_board = self.do_move(self.curr_board, move)

    def restore_to_round(self, round):
        round_dist = max(self.moves_and_pieces_in_round.keys()) - round
        board_ = self.curr_board
        if round_dist > round:  # deciding which way around to restore: from the beginning or the end
            # restore from end
            board_ = self.undo_last_n_moves(n=round, board=board_)
        else:
            # restore from beginning
            board_ = copy.deepcopy(self.initialBoard)
            for played_round in range(round):
                board_ = self.do_move(board_, self.moves_and_pieces_in_round[played_round][0])
        return board_

    def undo_last_n_moves(self, n, board):
        """
        Undo the last n moves in the memory. Return the updated board.
        :param board: numpy array
        :param n: int number of moves to undo
        :return: board
        """
        max_round = max(self.moves_and_pieces_in_round.keys())
        for k in range(n):
            (from_, to_), piece_from, piece_to = self.moves_and_pieces_in_round[max_round - k]
            board[from_] = piece_from
            board[to_] = piece_to
            piece_from.position = from_
            piece_to.position = to_
        return board

    def do_move(self, board, move):
        """
        :param move: tuple or array consisting of coordinates 'from' at 0 and 'to' at 1
        :param board: numpy array representing the board
        """
        from_ = move[0]
        to_ = move[1]
        if board[to_] is not None:  # Target field is not empty, then has to fight
            fight_outcome = Game.bm[board[from_].type, board[to_].type]
            if fight_outcome == 1:
                board[to_] = board[from_]
            elif fight_outcome == 0:
                board[to_] = None
        else:
            board[to_] = board[from_]
        board[from_] = None
        return board
