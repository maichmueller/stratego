import copy as cp
import game
import numpy as np
import torch
from matplotlib import pyplot as plt


import helpers
import pieces


class Env:
    """
    Environment superclass. Used for training the agent.
    """
    def __init__(self, agent0, agent1, board_size="small", fixed_setups=False):
        self.board_size = board_size
        self.agents = (agent0, agent1)
        if board_size == "small":
            self.types_available = np.array([0, 1] + [2]*3 + [3]*2 + [10] + [11]*2)
            self.obstacle_positions = [(2, 2)]
            self.game_dim = 5
        elif board_size == "medium":
            self.obstacle_positions = [(3, 1), (3, 5)]
            self.types_available = np.array([0, 1] + [2]*5 + [3]*3 + [4]*3 + [5]*2 + [6] + [10] + [11]*4)
            self.game_dim = 7
        else:
            self.obstacle_positions = [(4, 2), (5, 2), (4, 3), (5, 3), (4, 6), (5, 6), (4, 7), (5, 7)]
            self.types_available = np.array([0, 1] + [2]*8 + [3]*5 + [4]*4 + [5]*4 + [6]*4 +
                                            [7]*3 + [8]*2 + [9]*1 + [10] + [11]*6)
            self.game_dim = 10

        self.living_pieces = [[], []]  # team 0,  team 1
        self.dead_pieces = [[], []]

        if not fixed_setups:
            agent0.setup = self.draw_random_setup(self.types_available, 0, self.game_dim)
            agent1.setup = self.draw_random_setup(self.types_available, 1, self.game_dim)
        typecounter = [dict(), dict()]
        for idx, piece in np.ndenumerate(np.concatenate((agent0.setup, agent1.setup))):
            typecounter[piece.team][piece.type] = 0
            self.living_pieces[piece.team].append(piece)
            typecounter[piece.team][piece.type] += 1
            piece.version = typecounter[piece.team][piece.type]

        self.game_engine = game.Game(agent0, agent1, self.board_size)
        self.board = self.game_engine.board



        self.BATTLE_MATRIX = self.game_engine.BATTLE_MATRIX


        self.score = 0
        self.reward = 0
        self.steps = 0
        self.death_steps = None
        self.illegal_moves = 0

        # rewards (to be overridden by subclass environment)
        self.reward_illegal = 0  # punish illegal moves
        self.reward_step = 0  # negative reward per agent step
        self.reward_win = 0  # win game
        self.reward_loss = 0  # lose game
        self.reward_kill = 0  # kill enemy figure reward
        self.reward_die = 0  # lose to enemy figure

        self.move_count = 0

    def reset(self):  # resetting means freshly initializing
        self.__init__(agent0=self.agents[0], agent1=self.agents[1])

    def draw_random_setup(self, types_available, team, game_dim):
        """
        Draw a random setup from the set of types types_available after placing the flag
        somewhere in the last row of the board of the side of 'team', or behind the obstacle.
        :param types_available: list of types to draw from, integers
        :param team: boolean, 1 or 0 depending on the team
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

    def step(self, move=False):
        """
        Perform one step of the environment: agents in turn choose a move
        :param move: externally determined move to be performed by agent0 (useful for training)
        :return: reward accumulated in this step, boolean: if environment in terminal state, boolean: if agent0 won
        """
        self.reward = 0
        self.steps += 1
        done = False
        # first agent to act
        turn = self.game_engine.move_count % 2
        if turn == 0:
            if move is not False:
                agent0_outcome, agent1_outcome = self.game_engine.run_step(move)
            else:
                agent0_outcome, agent1_outcome = self.game_engine.run_step()
        else:
            agent0_outcome, agent1_outcome = self.game_engine.run_step()
        if agent0_outcome > 0 or agent1_outcome > 0:
            done = True
            self.reward += agent0_outcome
            self.score += self.reward
            self.move_count += 1
            return self.reward, done, agent0_outcome
        # second agent to act
        if turn == 0:
            if move is not False:
                agent0_outcome, agent1_outcome = self.game_engine.run_step(move)
            else:
                agent0_outcome, agent1_outcome = self.game_engine.run_step()
        else:
            agent0_outcome, agent1_outcome = self.game_engine.run_step()
        if agent0_outcome > 0 or agent1_outcome > 0:
            done = True
            self.reward += agent0_outcome
            self.score += self.reward
            self.move_count += 2
            return self.reward, done, agent0_outcome

        self.move_count += 2
        return self.reward, done, agent0_outcome

    def do_move(self, move, team):
        """
        Perfom the move provided by the team of 'team'
        :param move: tuple of tuple defining the 'from' position to the 'to' position
        :param team: boolean, 1 or 0
        :return: True for correct execution, False if not
        """
        if move is None:  # no move chosen (network)?
            return False
        if not helpers.is_legal_move(self.board, move):  # if move is illegal
            return False  # illegal move chosen
        other_team = (team + 1) % 2
        pos_from, pos_to = move
        piece_from = self.board[pos_from]
        piece_to = self.board[pos_to]

        # agents updating their board too
        for _agent in self.agents:
            _agent.do_move(move, true_gameplay=True)

        piece_from.has_moved = True
        if piece_to is not None:  # Target field is not empty, then has to fight
            fight_outcome = self.BATTLE_MATRIX[piece_from.type, piece_to.type]

            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return False
            elif fight_outcome == 1:  # attacker won
                self.update_board((pos_to, piece_from))
                self.update_board((pos_from, None))
                self.dead_pieces[other_team].append(piece_to)
                if team == 0:
                    self.reward += self.reward_kill
            elif fight_outcome == 0:  # both pieces died
                self.update_board((pos_to, None))
                self.update_board((pos_from, None))
                self.dead_pieces[team].append(piece_from)
                self.dead_pieces[other_team].append(piece_to)
            elif fight_outcome == -1:  # defender won
                self.update_board((pos_from, None))
                self.update_board((pos_to, piece_to))
                self.dead_pieces[team].append(piece_from)
                if team == 0:
                    self.reward += self.reward_die

        else:  # if there is no piece on the spot we want to move on, then we simply move
            self.update_board((pos_to, piece_from))
            self.update_board((pos_from, None))
            if team == 0:  # punish doing a step
                self.reward += self.reward_step

        return True

    def update_board(self, updated_piece):
        """
        Put the piece in updated_piece at the position given by updated_piece
        :param updated_piece: tuple of a position and a piece
        :return: None, change in-place
        """
        pos = updated_piece[0]
        piece = updated_piece[1]
        if piece is not None:
            piece.change_position(pos)  # adapt position for piece
        self.board[pos] = piece  # place piece on board position
        return

    def goal_test(self):
        """
        Check if the game is in a terminal state due to flag capture
        (note: in env.step it is already checked if there are still pieces to move)
        :return: (bool: is environment in a terminal state, bool: is it won (True) or lost (False) for player 0
        """
        # check whether the flag of team 1 has been captured
        for p in self.dead_pieces[1]:
            if p.type == 0:
                self.reward += self.reward_win
                return True, True
        # check whether the flag of team 0 has been captured
        for p in self.dead_pieces[0]:
            if p.type == 0:
                self.reward += self.reward_loss
                return True, False
        if self.death_steps is not None:
            if self.steps > self.death_steps:
                self.reward += self.reward_loss
                return True, False
        return False, False

    def show(self):
        fig = plt.figure(1)
        helpers.print_board(self.board)
        plt.title("Reward = {}".format(self.score))
        fig.canvas.draw()  # updates plot


