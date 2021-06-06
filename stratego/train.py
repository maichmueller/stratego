import torch.optim as optim

import agent
from stratego.algorithms import algorithms
import utils


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
        won_games = 0
        too_many_steps_games = 0
        while not done:
            state = env.agents[0].state_to_tensor()  # for the reinforcement agent convert board to state input
            action = env.agents[0].sample_action(state, 0.00)
            if action is not None:
                action = action[0, 0]  # action is unwrapped from the LongTensor
            move = env.agents[0].action_to_move(action)  # e.g. action = 1 -> move = ((0, 0), (0, 1))
            _, done, won = env.step(move)
            if show:
                env.show()
            if done:
                if won:
                    won_games += 1
                elif env.steps > 100:
                    too_many_steps_games += 1
                break
    print("Wins: {}/{}".format(won_games, n_runs))
    print("Number of cancelled games: {}".format(too_many_steps_games))



# hyperparameters

device = utils.get_device()
print("TORCH DEVICE USED: {}".format(device))
PLOT_FREQUENCY = 500
BATCH_SIZE = 1024  # for faster training take a smaller batch size, not too small as batchnorm will not work otherwise
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
agent1 = agent.RandomAgent(1)
# agent1 = agent.Random(1)
# agent1.model = agent0.model  # if want to train by self-play
env__ = algorithms.Trainer(agent0, agent1, False, "custom", [0, 1])
env_name = "stratego"

model = env__.agents[0].model  # optimize model of agent0
model = model.to(device)
optimizer = optim.Adam(model.parameters())
memory = utils.ReplayMemory(10000)

# model.load_state_dict(torch.load('./saved_models/{}_current.pkl'.format(env_name)))  # trained against Random
train(env__, num_episodes)
# model.load_state_dict(torch.load('./saved_models/{}.pkl'.format(env_name)))  # trained against Random

run_env(env__, 10000)

