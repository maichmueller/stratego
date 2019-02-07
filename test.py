import game
import agent
import coach
import numpy as np
from cythonized import utils

g = game.Game(agent.MiniMax(0), agent.Random(1), board_size='small')
actions, relation_dict = utils.action_rep.actions, utils.action_rep.act_piece_relation
while True:

    actions_mask = utils.get_actions_mask(g.state.board, g.move_count % 2,
                                          relation_dict,
                                          actions)
    lm = []
    for idx, legal in enumerate(actions_mask):
        if legal:
            lm.append(g.state.action_to_move(idx, g.move_count % 2))

    #print(lm)

    if utils.get_poss_moves(g.state.board, g.move_count % 2):
        move = lm[np.random.choice(len(lm))]
        print(move)
        r = g.run_step(move)
    else:
        r = 1
    if r != 404:
        print('Game done', flush=True)
        g.reset()
