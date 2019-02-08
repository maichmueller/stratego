import numpy as np
from scipy import spatial
from matplotlib import pyplot as plt

from collections import namedtuple
import random
from sklearn.manifold import TSNE
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import copy
import random


ID_SET = set(range(100000))


def set_id():
    global ID_SET
    id_ = random.sample(ID_SET, 1)
    ID_SET -= set(id_)
    return id_


class GameDefaults:
    def __init__(self, board_size='small'):
        self.board_size = board_size
        self.types_available = np.array([0, 1] + [2] * 3 + [3] * 2 + [10] + [11] * 2)
        self.obstacle_positions = [(2, 2)]
        self.game_dim = 5

    def set_board_size(self, board_size):
        if board_size == self.board_size:
            return

        self.board_size = board_size
        if self.board_size in ["small", 5]:
            self.types_available = np.array([0, 1] + [2] * 3 + [3] * 2 + [10] + [11] * 2)
            self.obstacle_positions = [(2, 2)]
            self.game_dim = 5
        elif self.board_size in ["medium", 7]:
            self.obstacle_positions = [(3, 1), (3, 5)]
            self.types_available = np.array([0, 1] + [2] * 5 + [3] * 3 + [4] * 3 + [5] * 2 + [6] + [10] + [11] * 4)
            self.game_dim = 7
        elif self.board_size in ["big", 10]:
            self.obstacle_positions = [(4, 2), (5, 2), (4, 3), (5, 3), (4, 6), (5, 6), (4, 7), (5, 7)]
            self.types_available = np.array([0, 1] + [2] * 8 + [3] * 5 + [4] * 4 + [5] * 4 + [6] * 4 +
                                            [7] * 3 + [8] * 2 + [9] * 1 + [10] + [11] * 6)
            self.game_dim = 10

    def get_game_specs(self):
        return self.obstacle_positions, self.types_available, self.game_dim


GameDef = GameDefaults()


def get_bm():
    bm = dict()
    for i in range(1, 11):
        for j in range(1, 11):
            if i < j:
                bm[i, j] = -1
                bm[j, i] = 1
            elif i == j:
                bm[i, i] = 0
        bm[i, 0] = 1
        if i == 3:
            bm[i, 11] = 1
        else:
            bm[i, 11] = -1
    bm[1, 10] = 1
    return bm


def is_legal_move(board, move_to_check):
    """
    :param move_to_check: array/tuple with the coordinates of the position from and to
    :return: True if move is a legal move, False if not
    """
    if move_to_check is None:
        return False
    pos_before = move_to_check[0]
    pos_after = move_to_check[1]

    for x in (pos_before[0], pos_before[1], pos_after[0], pos_after[1]):
        if not -1 < x < board.shape[0]:
            return False

    # piece to move is not at this position
    if board[pos_before] is None:
        return False

    if not board[pos_after] is None:
        if board[pos_after].team == board[pos_before].team:
            return False  # cant fight own pieces
        if board[pos_after].type == 99:
            return False  # cant fight obstacles

    move_dist = spatial.distance.cityblock(pos_before, pos_after)
    if move_dist > 1:
        if pos_after[0] == pos_before[0]:
            dist_sign = int(np.sign(pos_after[1] - pos_before[1]))
            for k in list(range(pos_before[1] + dist_sign, pos_after[1], int(dist_sign))):
                if board[(pos_before[0], k)] is not None:
                    return False  # pieces in the way of the move
        else:
            dist_sign = int(np.sign(pos_after[0] - pos_before[0]))
            for k in range(pos_before[0] + dist_sign, pos_after[0], int(dist_sign)):
                if board[(k, pos_before[1])] is not None:
                    return False  # pieces in the way of the move

    return True


class ActionRep:
    def __init__(self):
        self.actors = None
        self.actions = None
        self.act_piece_relation = None
        self.action_dim = None
        self.game_dim = GameDef.get_game_specs()[2]
        self.build_action_rep()

    def build_action_rep(self, force=False):
        if force or any([x is None for x in (self.actors, self.actions, self.act_piece_relation)]):
            action_rep_pieces = []
            action_rep_moves = []
            action_rep_dict = dict()
            for type_ in sorted(GameDef.get_game_specs()[1]):
                version = 1
                type_v = str(type_) + "_" + str(version)
                while type_v in action_rep_pieces:
                    version += 1
                    type_v = type_v[:-1] + str(version)
                if type_ in [0, 11]:
                    continue
                elif type_ == 2:
                    actions = [(i, 0) for i in range(1, self.game_dim)] + \
                              [(0, i) for i in range(1, self.game_dim)] + \
                              [(-i, 0) for i in range(1, self.game_dim)] + \
                              [(0, -i) for i in range(1, self.game_dim)]
                    len_acts = len(actions)
                    len_acts_sofar = len(action_rep_moves)
                    action_rep_dict[type_v] = list(range(len_acts_sofar, len_acts_sofar + len_acts))
                    action_rep_pieces += [type_v] * len_acts
                    action_rep_moves += actions
                else:
                    actions = [(1, 0),
                               (0, 1),
                               (-1, 0),
                               (0, -1)]
                    action_rep_dict[type_v] = list(range(len(action_rep_moves), len(action_rep_moves) + 4))
                    action_rep_pieces += [type_v] * 4
                    action_rep_moves += actions
            self.act_piece_relation = action_rep_dict
            self.actions = tuple(action_rep_moves)
            self.actors = tuple(action_rep_pieces)
            self.action_dim = len(action_rep_moves)


action_rep = ActionRep()


def get_actions_mask(board, team, action_rep_dict, action_rep_moves):
    """
    :return: List of possible actions for agent of team 'team'
    """
    game_dim = board.shape[0]
    actions_mask = np.zeros(len(action_rep_moves), dtype=int)
    for pos, piece in np.ndenumerate(board):
        if piece is not None and piece.team == team and piece.can_move:  # board position has a piece on it
            # get the index range of this piece in the moves list
            p_range = np.array(action_rep_dict[str(piece.type) + "_" + str(piece.version)])
            # get the associated moves to this piece
            p_moves = [action_rep_moves[i] for i in p_range]
            if piece.type == 2:
                poss_fields = [(pos[0] + i, pos[1]) for i in range(1, game_dim - pos[0])] +\
                              [(pos[0], pos[1] + i) for i in range(1, game_dim - pos[1])] + \
                              [(pos[0] - i, pos[1]) for i in range(1, pos[0]+1)] +\
                              [(pos[0], pos[1] - i) for i in range(1, pos[1]+1)]

                for pos_to in poss_fields:
                    move = (pos, pos_to)
                    if is_legal_move(board, move):
                        base_move = (pos_to[0] - pos[0], pos_to[1] - pos[1])
                        base_move_idx = p_moves.index(base_move)
                        actions_mask[p_range.min() + base_move_idx] = 1

            else:
                poss_fields = [(pos[0]+1, pos[1]),
                               (pos[0], pos[1]+1),
                               (pos[0]-1, pos[1]),
                               (pos[0], pos[1]-1)]
                for pos_to in poss_fields:
                    move = (pos, pos_to)
                    if is_legal_move(board, move):
                        base_move = (pos_to[0] - pos[0], pos_to[1] - pos[1])
                        base_move_idx = p_moves.index(base_move)
                        actions_mask[p_range.min() + base_move_idx] = 1
    return actions_mask


def print_board(board, same_figure=True, block=False):
    """
    Plots a board object in a pyplot figure
    :param same_figure: Should the plot be in the same figure?
    """
    game_dim = board.shape[0]
    board = copy.deepcopy(board)  # ensure to not accidentally change input
    # plt.interactive(False)  # make plot stay? true: close plot, false: keep plot
    if same_figure:
        plt.figure(1)  # needs to be one
    else:
        plt.figure()
    plt.clf()
    # layout = np.add.outer(range(game_dim), range(game_dim)) % 2  # chess-pattern board
    layout = np.zeros((game_dim, game_dim))
    plt.imshow(layout, cmap=plt.cm.magma, alpha=.0, interpolation='nearest')  # plot board

    # plot lines separating each cell for visualization
    for i in range(game_dim + 1):
        plt.plot([i-.5, i-.5], [-.5, game_dim-.5], color='k', linestyle='-', linewidth=1)
        plt.plot([-.5, game_dim-.5], [i-.5, i-.5], color='k', linestyle='-', linewidth=1)

    # go through all board positions and print the respective markers
    for pos in ((i, j) for i in range(game_dim) for j in range(game_dim)):
        piece = board[pos]  # select piece on respective board position
        # decide which marker type to use for piece
        if piece is not None:
            # piece.hidden = False  # omniscient view

            if piece.team == 1:
                color = 'b'  # blue: player 1
            elif piece.team == 0:
                color = 'r'  # red: player 0
            else:
                color = 'k'  # black: obstacle

            if piece.can_move:
                form = 'o'  # circle: for movable
            else:
                form = 's'  # square: either immovable or unknown piece
            if piece.type == 0:
                form = 'X'  # cross: flag

            piece_marker = ''.join(('-', color, form))
            alpha = 0.3 if piece.hidden else 1
            plt.plot(pos[1], pos[0], piece_marker, markersize=37, alpha=alpha)  # plot marker
            # piece type written on marker center
            plt.annotate(str(piece), xy=(pos[1], pos[0]), color='w', size=20, ha="center", va="center")
    # invert y makes numbering more natural; puts agent 1 on bottom, 0 on top !
    plt.gca().invert_yaxis()
    plt.pause(1)
    plt.show(block=block)


def get_poss_moves(board, team):
    """
    :return: List of possible actions for agent of team 'team'
    """
    game_dim = board.shape[0]
    actions_possible = []
    for pos, piece in np.ndenumerate(board):
        if piece is not None:  # board position has a piece on it
            if piece.team == team:
                # check which moves are possible
                if piece.can_move:
                    if piece.type == 2:
                        poss_fields = [(pos[0] + i, pos[1]) for i in range(1, game_dim - pos[0])] +\
                                      [(pos[0], pos[1] + i) for i in range(1, game_dim - pos[1])] + \
                                      [(pos[0] - i, pos[1]) for i in range(1, pos[0]+1)] +\
                                      [(pos[0], pos[1] - i) for i in range(1, pos[1]+1)]
                        for pos_to in poss_fields:
                            move = (pos, pos_to)
                            if is_legal_move(board, move):
                                actions_possible.append(move)
                    else:
                        poss_fields = [(pos[0]+1, pos[1]),
                                       (pos[0], pos[1]+1),
                                       (pos[0]-1, pos[1]),
                                       (pos[0], pos[1]-1)]
                        for pos_to in poss_fields:
                            move = (pos, pos_to)
                            if is_legal_move(board, move):
                                actions_possible.append(move)
    return actions_possible


def plot_scores(episode_scores, n_smooth):
    """
    Plots the scores (accumulated reward over one episode) for a RL agent
    :param episode_scores: list of the episode scores
    :param n_smooth: averaged over this number of episodes
    :return: plot
    """
    plt.figure(2)
    plt.clf()
    scores_t = torch.FloatTensor(episode_scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    average = [0]
    if len(scores_t) >= n_smooth:
        means = scores_t.unfold(0, n_smooth, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(n_smooth-1), means))
        average = means.numpy()
        plt.plot(average)
    plt.title('Average Score over last {} Episodes: {}'.format(n_smooth, int(average[-1]*10)/10))
    #plt.pause(0.001)  # pause a bit so that plots are updated


def plot_stats(curr_average, episode_won, n_smooth, plot_freq):
    """
    Plots the winning/losing ratio
    :param episode_scores:
    :param n_smooth:
    :return:
    """
    plt.figure(2)
    plt.clf()
    scores = np.array(episode_won)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    if len(scores) >= n_smooth:
        mean = np.mean(scores[-n_smooth:])
        curr_average.append(mean)
        x = [n_smooth+plot_freq*i for i in range(len(curr_average))]  # x axis values
        xi = [i for i in range(0, len(x))]  # ticks for each x point
        selection_val = int(np.floor(len(xi)/5)) + 1
        xi = [tick for tick in xi if tick % selection_val == 0]
        x = [tick for idx, tick in enumerate(x) if idx % selection_val == 0]
        plt.xticks(xi, x)
        plt.plot(curr_average)
        plt.title('Average Win Percentage over last {} Episodes: {}'.format(n_smooth, int(curr_average[-1]*100)/100))
        plt.pause(0.003)  # pause a bit so that plots are updated
    return curr_average


def plot_stats_all(episode_won, end_episode):
    """
    Plots the winning/losing ratio
    :param episode_scores:
    :param n_smooth:
    :return:
    """
    plt.figure(3)
    plt.clf()
    scores_t = torch.FloatTensor(episode_won)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    average = [0]
    for i in range(1, end_episode+1):
        means = scores_t.unfold(0, i, 1).mean(1).view(-1)
        average.append(list(means.numpy())[0])
    plt.plot(average)
    plt.title('Average Win Percentage over last {} Episodes: {}'.format(end_episode, int(average[-1]*100)/100))
    plt.pause(0.001)  # pause a bit so that plots are updated


class ReplayMemory(object):
    """
    Stores a state-transition (s, a, s', r) quadruple
    for approximating Q-values with q(s, a) <- r + gamma * max_a' q(s', a') updates
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def visualize_features(n_points, environment, env_name):
    """
    Visualize with t-SNE the models representation of a board state
    :param n_points: number of points in t-SNE plot
    :param env_name: environment name for plotting
    :return: plot of t-SNE and plot of some board states
    """
    boards = []
    states = []
    model = environment.agents[0].model
    interrupt = False

    print("Acquiring features")
    for i in range(n_points):
        environment.reset()
        done = False

        while not done:
            _, done, won = environment.step()

            board = copy.deepcopy(environment.board)
            state = environment.agents[0].board_to_state()
            boards.append(board)
            states.append(state)
            if environment.steps > 20:  # break loops to obtain more diverse feature space
                break
            if len(states) >= n_points:  # interrupt simulation if enough states
                interrupt = True
                break
        if interrupt:
            break

    states = Variable(torch.cat(states))
    features = model.extract_features(states)
    action_values = F.sigmoid(model.lin_final(features))
    features = features.data.numpy()
    state_values = action_values.data.numpy().max(1)

    print("Computing t-SNE embedding")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(features)

    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)  # scale to interval (0, 1)

    # print out 10 randomly chosen board states with positions and state-values
    # and mark choice in embedding
    choice = np.random.choice(len(boards), 10, replace=False)
    for i, c in enumerate(choice):
        print(i, c)
        print(X_tsne[c])
        print(state_values[c])
        print_board(boards[c], same_figure=False)
        # plt.title("state value: {}, position: {}".format(state_values[c], X_tsne[c]))
        plt.savefig('{}{}.png'.format(env_name, i))

    def plot_embedding(features, values, choice):
        """
        Plot an embedding
        :param features: vectors to be plotted
        :param values: value between 0 and 1 for respective color (e.g. state-value)
        :return: plot
        """
        fig, ax = plt.subplots()
        mymap = plt.cm.get_cmap('Spectral')
        sm = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []  # fake array for scalar mappable urrgh..
        for i in range(features.shape[0]):
            plt.plot(features[i, 0], features[i, 1], '.', color=mymap(values[i]))
        cb = plt.colorbar(sm)
        cb.set_label('Q-Values')
        for i in range(features.shape[0]):  # overwrite old plotted points
            if i in choice:
                plt.plot(features[i, 0], features[i, 1], 'o', color='k')
        plt.xticks([]), plt.yticks([])
        plt.title("t-SNE of Board Evaluator Features")
        plt.show(block=False)

    print("Plotting t-SNE embedding")
    plot_embedding(X_tsne, state_values, choice)
    plt.savefig('{}-tsne.png'.format(env_name))


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
