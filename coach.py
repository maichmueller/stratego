from copy import deepcopy
from game import Game
import agent
import numpy as np
from matplotlib import pyplot as plt

from cythonized.mcts import MCTS, utils

from cythonized.utils import AverageMeter
from progressBar.progress.bar import Bar
import time
from collections import deque
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle
from arena import Arena
import models


class Coach:
    def __init__(self, student, num_iters=100, **kwargs):
        # super().__init__(*args, **kwargs)

        self.num_iters = num_iters
        self.win_frac = 0.55
        self.num_episodes = 100
        self.mcts_sims = 1000
        self.num_iters_trainex_hist = 7
        self.model_folder = './saved_models/'
        self.temp_thresh = 1000

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
        self.train_expls_hist = []   # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skip_first_self_play = False  # can be overriden in loadTrainExamples()

    def exec_ep(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (board,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        state = self.game.state
        ep_step = 0

        def invert_move(move):
            from_, to_ = move
            return ((self.game.game_dim-1 - from_[0], self.game.game_dim-1 - from_[1]),
                    (self.game.game_dim-1 - to_[0], self.game.game_dim-1 - to_[1]))

        while True:
            ep_step += 1
            utils.print_board(self.game.state.board)
            temp = int(ep_step < self.temp_thresh)

            turn = self.game.move_count % 2
            pi = self.mcts.get_action_prob(state, player=turn, temp=temp)

            action = np.random.choice(len(pi), p=pi)
            move = self.game.state.action_to_move(action, 0, force=True)
            print(move)
            moves = []
            for i, a in enumerate(pi):
                if a > 0:
                    moves.append(self.game.state.action_to_move(i, 0, force=True))
            # print(moves)
            self.game.state.force_canonical(0)
            if turn == 1:
                move = invert_move(move)
            self.game.run_step(move=move)

            r = self.game.state.is_terminal()
            if r == 0:
                r = self.reward_draw

            if r != 404:
                return self.game.state.board, pi, r

    def learn(self):
        """
        Performs num_iters iterations with num_episodes episodes of self-play in each
        iteration. After every iteration, it retrains the neural network with
        examples in train_examples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.num_iters + 1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iter_train_expls = deque([])

                eps_time = AverageMeter()
                bar = Bar(desc='Self Play', total=self.num_episodes)
                end = time.time()

                for eps in range(self.num_episodes):
                    self.mcts = MCTS(self.game, self.nnet)  # reset search tree
                    iter_train_expls += self.exec_ep()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = f'({eps+1}/{self.num_episodes}) ' \
                        f'Eps Time: {eps_time.avg:.3f}s | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:}'
                    bar.next()
                bar.finish()

                # save the iteration examples to the history
                self.train_expls_hist.append(iter_train_expls)

            if len(self.train_expls_hist) > self.num_iters_trainex_hist:
                print("len(train_examples_history) =", len(self.train_expls_hist),
                      " => remove the oldest train_examples")
                self.train_expls_hist.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.save_train_examples(i - 1)

            # shuffle examlpes before training
            train_examples = []
            for e in self.train_expls_hist:
                train_examples.extend(e)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.model_folder, filename='temp.pth.tar')
            self.opp_net.load_checkpoint(folder=self.model_folder, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.opp_net)

            self.nnet.train(train_examples)
            nmcts = MCTS(self.game, self.nnet)

            print('PITTING AGAINST PREVIOUS VERSION')
            test_ag_0 = agent.Agent(0)
            test_ag_0.decide_move = lambda x: np.argmax(nmcts.get_action_prob(x, temp=0))
            test_ag_1 = agent.Agent(1)
            test_ag_1.decide_move = lambda x: np.argmax(pmcts.get_action_prob(x, temp=0))
            arena = Arena(test_ag_0, test_ag_1, self.game.board_size)
            ag_0_wins, ag_1_wins, draws = arena.pit(num_sims=self.num_iters)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (ag_0_wins, ag_1_wins, draws))
            if ag_0_wins + ag_1_wins > 0 and float(ag_0_wins) / (ag_0_wins + ag_1_wins) < self.win_frac:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.model_folder, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
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
                self.train_expls_hist = Unpickler(f).load()
            f.close()
            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True


if __name__ == '__main__':
    coach = Coach(agent.AlphaZero(0), board_size='small')
    coach.learn()