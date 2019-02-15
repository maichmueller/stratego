from copy import deepcopy
from game import Game
import agent
import numpy as np
from matplotlib import pyplot as plt

from cythonized.mcts import MCTS, utils

from cythonized.utils import AverageMeter
from progressBar.progress.bar import Bar
import time
from collections import deque, namedtuple
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle
from arena import Arena
import models
from tqdm.auto import tqdm


Episode = namedtuple('Episode', 'board pi v')


class Coach:
    def __init__(self, student, num_iterations=100, num_episodes=100,
                 num_iters_trainexample_history=10000, win_frac=0.55,
                 mcts_simulations=100, exploration_rate=10, **kwargs):
        # super().__init__(*args, **kwargs)

        self.num_iters = num_iterations
        self.win_frac = win_frac
        self.num_episodes = num_episodes
        self.mcts_sims = mcts_simulations
        self.num_iters_trainex_hist = num_iters_trainexample_history
        self.model_folder = './checkpoints/'
        self.temp_thresh = exploration_rate

        self.score = 0
        self.reward = 0
        self.steps = 0
        self.death_steps = None
        self.illegal_moves = 0

        self.reward_illegal = 0  # punish illegal moves
        self.reward_step = 0  # negative reward per agent step
        self.reward_win = 1  # win game
        self.reward_loss = -1  # lose game
        self.reward_kill = 0  # kill enemy figure reward
        self.reward_die = 0  # lose to enemy figure
        self.reward_draw = 0

        self.game = Game(agent0=student, agent1=deepcopy(student), **kwargs)
        self.nnet = student.model
        self.opp_net = self.game.agents[1].model  # the competitor network
        self.mcts = MCTS(self.game, self.nnet)
        self.train_expls_hist = []   # history of examples from num_iters_for_train_examples_history latest iterations
        self.skip_first_self_play = False  # can be overriden in load_train_examples()

    def exec_ep(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a expl_rate=1 if episodeStep < tempThreshold, and thereafter
        uses expl_rate=0.
        Returns:
            trainExamples: a list of examples of the form (board,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        self.game.reset()
        state = deepcopy(self.game.state)
        ep_step = 0

        def invert_move(m):
            from_, to_ = m
            return ((self.game.game_dim-1 - from_[0], self.game.game_dim-1 - from_[1]),
                    (self.game.game_dim-1 - to_[0], self.game.game_dim-1 - to_[1]))

        while True:
            ep_step += 1
            # utils.print_board(self.game.state.board)
            expl_rate = int(ep_step < self.temp_thresh)

            turn = state.move_count % 2

            pi = self.mcts.get_action_prob(state, player=turn, expl_rate=expl_rate)
            if isinstance(pi, int):
                state.force_canonical(0)
                r = state.is_terminal(force=True)
                pi = None
                return state.board, pi, r

            action = np.random.choice(len(pi), p=pi)
            state.force_canonical(player=turn)
            move = state.action_to_move(action, 0, force=True)
            state.force_canonical(0)

            if turn == 1:
                move = invert_move(move)

            state.do_move(move=move)
            r = state.is_terminal(force=True)
            # utils.print_board(state.board)
            # print(r)
            if r != 404:
                return state.board, pi, r

    def teach(self, from_prev_examples=False):
        """
        Performs num_iters iterations with num_episodes episodes of self-play in each
        iteration. After every iteration, it retrains the neural network with
        examples in train_examples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        if from_prev_examples:
            i = 0
            while True:
                if os.path.isfile(self.model_folder + f'checkpoint_{i}.pth.tar' + ".examples"):
                    checkpoint = f'checkpoint_{i}.pth.tar'
                    self.load_train_examples(checkpoint)
                    i += 1
                else:
                    break
            if os.path.isfile(self.model_folder + f'best.pth.tar'):
                self.nnet.load_checkpoint(self.model_folder, f'best.pth.tar')

        for i in range(1, self.num_iters + 1):
            # bookkeeping

            print('\n------ITER ' + str(i) + '------')

            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iter_train_expls = []

                for _ in tqdm(range(self.num_episodes)):
                    # bookkeeping + plot progress through tqdm
                    # reset search tree
                    self.mcts = MCTS(self.game, self.nnet, num_mcts_sims=self.mcts_sims)
                    ep = Episode(*self.exec_ep())
                    iter_train_expls.append(ep)
                    self.game.reset()

                # save the iteration examples to the history
                self.train_expls_hist += iter_train_expls

            diff_hist_len = len(self.train_expls_hist) - self.num_iters_trainex_hist
            if diff_hist_len > 0:
                print("len(train_examples_history) =", len(self.train_expls_hist),
                      f" => remove the oldest {diff_hist_len} train_examples")
                self.train_expls_hist = self.train_expls_hist[diff_hist_len:]
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.save_train_examples(i - 1)

            # shuffle examples before training
            train_examples = []
            for epi in self.train_expls_hist:
                train_examples += [Episode(self.game.agents[0].board_to_state(epi.board),
                                         epi.pi, epi.v)]
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.model_folder, filename='temp.pth.tar')
            self.opp_net.load_checkpoint(folder=self.model_folder, filename='temp.pth.tar')
            # pmcts = MCTS(self.game, self.opp_net)

            self.nnet.train(train_examples, 100)
            # nmcts = MCTS(self.game, self.nnet)

            print('\nPITTING AGAINST PREVIOUS VERSION')
            test_ag_0 = agent.AlphaZero(0, low_train=True)
            test_ag_0.model.load_checkpoint(folder=self.model_folder, filename='temp.pth.tar')
            # test_ag_0.decide_move = lambda x: np.argmax(nmcts.get_action_prob(x, expl_rate=0))
            test_ag_1 = agent.AlphaZero(1, low_train=True)
            test_ag_1.model.load_checkpoint(folder=self.model_folder, filename='temp.pth.tar')
            # test_ag_1.decide_move = lambda x: np.argmax(pmcts.get_action_prob(x, expl_rate=0))
            arena = Arena(test_ag_0, test_ag_1, board_size=self.game.board_size)
            ag_0_wins, ag_1_wins, draws = arena.pit(num_sims=self.num_iters)

            print(f'Wins / losses of new model: {ag_0_wins} / {ag_1_wins } '
                  f'({round(ag_0_wins / (ag_1_wins + ag_0_wins), 3)}%) | draws: {draws}')
            if ag_0_wins + ag_1_wins > 0 and float(ag_0_wins) / (ag_0_wins + ag_1_wins) < self.win_frac:
                print('REJECTING NEW MODEL\n')
                self.nnet.load_checkpoint(folder=self.model_folder, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL\n')
                self.nnet.save_checkpoint(folder=self.model_folder, filename=f'checkpoint_{i}.pth.tar')
                self.nnet.save_checkpoint(folder=self.model_folder, filename='best.pth.tar')

    def save_train_examples(self, iteration):
        folder = self.model_folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f'checkpoint_{iteration}.pth.tar' + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_expls_hist)
        f.close()

    def load_train_examples(self, examples_fname):
        model_fname = os.path.join(self.model_folder, examples_fname)
        examples_file = model_fname + ".examples"
        if not os.path.isfile(examples_file):
            print(examples_file)
            r = input("File with train examples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with train_examples found. Reading it.")
            with open(examples_file, "rb") as f:
                self.train_expls_hist += Unpickler(f).load()
            f.close()
            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True


if __name__ == '__main__':
    c = Coach(agent.AlphaZero(0),
              num_episodes=20,
              mcts_simulations=100,
              board_size='small')
    c.teach(from_prev_examples=True)
