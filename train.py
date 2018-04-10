import math
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import agent
import env
import helpers
import pickle
import copy

# training of DQN network code is adjusted from Adam Paszke
# http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

def optimize_model():
    """
    Sample batch from memory of environment transitions and train network to fit the
    temporal difference TD(0) Q-value approximation
    """
    if len(memory) < BATCH_SIZE:
            return  # not optimizing for not enough memory
    model.train()
    transitions = memory.sample(BATCH_SIZE)  # sample memories batch
    batch = helpers.Transition(*zip(*transitions))  # transpose the batch

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))

    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.FloatTensor))  # zero for terminal states
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]  # what would the model predict
    next_state_values.volatile = False  # requires_grad = False to not mess with loss
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # compute the expected Q values

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)  # compute Huber loss

    # optimize network
    optimizer.zero_grad()  # optimize towards expected q-values
    loss.backward()
    for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
    optimizer.step()


def run_env(env, n_runs=100, show=True):
    """
    Plots simulated games in an environment for visualization
    :param env: environment to be run
    :param n_runs: how many episodes should be run
    :return: plot of each step in the environment
    """
    for i in range(n_runs):
        env.reset()
        if show:
            env.show()
        done = False
        outcome_sequence = []
        while not done:
            state = env.agents[0].board_to_state()  # for the reinforcement agent convert board to state input
            action = env.agents[0].select_action(state, 0.00)
            action = action[0, 0]  # action is unwrapped from the LongTensor
            move = env.agents[0].action_to_move(action)  # e.g. action = 1 -> move = ((0, 0), (0, 1))
            _, done, won = env.step(move)
            if show:
                env.show()
            if done:
                if won:
                    outcome_sequence.append(1)
                    print("Won! Overall: {}/{}".format(sum(outcome_sequence), n_runs))
                elif env.steps > 100:
                    print("Too many steps")
                else:
                    print("Lost! Overall: {}/{}".format(sum(outcome_sequence), n_runs))
                break


def train(env_, num_episodes):
    """
    Trains a reinforcement agent, acting according to the agents model
    or randomly (with exponentially decaying probability p_random)
    Each transition (state, action, next_state, reward) is stored in a memory.
    Each step in the environment is followed by a learning phase:
        a batch of memories is used to optimize the network model
    :param env_: training environement
    :param num_episodes: number of training episodes
    :return:
    """
    episode_scores = []  # score = total reward
    episode_won = [0]  # win-ratio win = 1 loss = -1
    averages = []
    best_winratio = 0.5
    for i_episode in range(num_episodes):
            env_.reset()  # initialize environment
            state = env_.agents[0].board_to_state()  # initialize state
            if (i_episode+1) % 1000 == 0:
                test_agent0 = agent.Stratego(0)
                test_agent0.model = agent0.model
                test_agent1 = agent.Random(1)
                test_env = env.Env(test_agent0, test_agent1)
                run_env(test_env, show=False)
                print("\n")
            while True:
                    # act in environment
                    p_random = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i_episode / EPS_DECAY)
                    action = env_.agents[0].select_action(state, p_random)  # random action with p_random
                    if action is not None:
                        move = env_.agents[0].action_to_move(action[0, 0])
                    else:
                        move = None
                    # environment step for action
                    reward_value, done, won = env_.step(move)

                    # env.show()
                    # if VERBOSE > 2:
                    #         print(action[0, 0], reward_value)
                    reward = torch.FloatTensor([reward_value]).cuda() if use_cuda else torch.FloatTensor([reward_value])

                    # save transition as memory and optimize model
                    if done:  # if terminal state
                        next_state = None
                        if won > 0:
                            won = 1
                        else:
                            won = 0
                    else:
                        next_state = env_.agents[0].board_to_state()

                    if move is not None:
                        memory.push(state, action, next_state, reward)  # store the transition in memory
                    state = next_state  # move to the next state
                    optimize_model()  # one step of optimization of target network

                    if done:
                        # after each episode print stats
                        if VERBOSE > 0:
                            print("Episode {}/{}".format(i_episode, num_episodes))
                            print("Score: {}%".format(100*round(sum(episode_won)/len(episode_won), ndigits=3)))
                            print("Won: {}".format(won))
                            print("Noise: {}".format(p_random))
                            print("Illegal: {}/{}\n".format(env_.illegal_moves, env_.steps))
                        episode_scores.append(env_.score)
                        episode_won.append(won)
                        if VERBOSE > 1:
                            if i_episode % PLOT_FREQUENCY == 0:
                                print("Episode {}/{}".format(i_episode, num_episodes))
                                # helpers.plot_scores(episode_scores, N_SMOOTH)  # takes run time
                                averages = helpers.plot_stats(averages, episode_won, N_SMOOTH, PLOT_FREQUENCY) # takes run time
                                torch.save(model.state_dict(), './saved_models/{}_current.pkl'.format(env_name))
                                if averages:
                                    if averages[-1] > best_winratio:
                                        best_winratio = averages[-1]
                                        print("Best win ratio: {}".format(np.round(best_winratio, 2)))
                                        torch.save(model.state_dict(), './saved_models/{}_best.pkl'.format(env_name))
                                pickle.dump(averages, open("{}-averages.p".format(env_name), "wb"))
                                pickle.dump(episode_won, open("{}-episode_won.p".format(env_name), "wb"))

                        break
            if i_episode % 500 == 2:
                    if VERBOSE > 2:
                            run_env(env_, 1)


# hyperparameters

use_cuda = torch.cuda.is_available()
PLOT_FREQUENCY = 500
BATCH_SIZE = 256*10  # for faster training take a smaller batch size, not too small as batchnorm will not work otherwise
GAMMA = 0.9  # already favors reaching goal faster, no need for reward_step, the lower GAMMA the faster
EPS_START = 0.9  # for unstable models take higher randomness first
EPS_END = 0.01
EPS_DECAY = 2000
N_SMOOTH = 500  # plotting scores averaged over this number of episodes
VERBOSE = 1  # level of printed output verbosity:
                # 1: plot averaged episode stats
                # 2: also print actions taken and rewards
                # 3: every 100 episodes run_env()
                # also helpful sometimes: printing probabilities in "select_action" function of agent

num_episodes = 100000  # training for how many episodes
agent0 = agent.Stratego(0)
agent1 = agent.Stratego(1)
# agent1 = agent.Random(1)
agent1.model = agent0.model  # if want to train by self-play
env__ = env.Env(agent0, agent1)
env_name = "stratego"

model = env__.agents[0].model  # optimize model of agent0
model = model.cuda() if use_cuda else model
optimizer = optim.Adam(model.parameters())
memory = helpers.ReplayMemory(10000)

# model.load_state_dict(torch.load('./saved_models/{}_current.pkl'.format(env_name)))  # trained against Random
train(env__, num_episodes)
# model.load_state_dict(torch.load('./saved_models/{}.pkl'.format(env_name)))  # trained against Random

run_env(env, 10000)

