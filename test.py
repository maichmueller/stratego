import agent
import arena
import os

#g = game.Game(agent.MiniMax(0), agent.Random(1), board_size='small')
#actions, relation_dict = utils.action_rep.actions, utils.action_rep.act_piece_relation
# ar = arena.Arena(agent.AlphaZero(0, low_train=True), agent.Random(1))
ar = arena.Arena(agent.Random(0), agent.MiniMax(1))
# if os.path.isfile('./checkpoints/' + f'best.pth.tar'):
#     ar.agent_0.model.load_checkpoint('./checkpoints/', f'best.pth.tar')
# if os.path.isfile('./checkpoints/' + f'best.pth.tar'):
#    ar.agent_1.model.load_checkpoint('./checkpoints/', f'best.pth.tar')

ar.pit(1000, show_game=False)