import game
import agent
import coach
import numpy as np
from cythonized import utils
import random

g = game.Game(agent.MiniMax(0), agent.Random(1), board_size='small')
actions, relation_dict = utils.action_rep.actions, utils.action_rep.act_piece_relation
while True:
    turn = g.move_count % 2
    actions_mask = utils.get_actions_mask(g.state.board, turn,
                                          relation_dict,
                                          actions)
    poss_moves = utils.get_poss_moves(g.state.board, turn)
    lm = []
    for idx, legal in enumerate(actions_mask):
        if legal:
            lm.append(g.state.action_to_move(idx, turn))
    lm = sorted(lm)
    intersec = list(set(lm) & set(sorted(poss_moves)))
    print(len(lm) == len(intersec))

    if poss_moves:
        move = lm[np.random.choice(np.array(lm).shape[0])]
        #print(move)
        r = g.run_step(move)
    else:
        r = 1
    if r != 404:
        print('Game done', flush=True)
        g.reset()
