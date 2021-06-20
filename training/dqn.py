from stratego import *

import torch


if __name__ == '__main__':
    DQN(Game(DQNAgent(0), DQNAgent(1)))
