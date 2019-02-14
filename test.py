import game
import agent
import coach
import numpy as np
from cythonized import utils
import random
import arena

#g = game.Game(agent.MiniMax(0), agent.Random(1), board_size='small')
#actions, relation_dict = utils.action_rep.actions, utils.action_rep.act_piece_relation
ar = arena.Arena(agent.AlphaZero(0, low_train=True), agent.AlphaZero(1, low_train=True))

ar.pit(100)