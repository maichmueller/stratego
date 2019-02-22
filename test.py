import game
import agent
import coach
import numpy as np
from cythonized import utils
import random
import arena
import os

#g = game.Game(agent.MiniMax(0), agent.Random(1), board_size='small')
#actions, relation_dict = utils.action_rep.actions, utils.action_rep.act_piece_relation
ar = arena.Arena(agent.AlphaZero(0, low_train=True), agent.AlphaZero(1, low_train=True))

if os.path.isfile('./checkpoints/' + f'best.pth.tar'):
    ar.agent_0.model.load_checkpoint('./checkpoints/', f'best.pth.tar')
if os.path.isfile('./checkpoints/' + f'best.pth.tar'):
    ar.agent_1.model.load_checkpoint('./checkpoints/', f'best.pth.tar')

ar.pit(10000, show_game=True)