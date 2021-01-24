from enum import Enum


class RewardToken(Enum):
    illegal = 0  # punish illegal moves
    step = 1  # negative reward per agent step
    win = 2  # win game
    loss = 3  # lose game
    kill = 4  # kill enemy piece reward
    die = 5  # lose to enemy piece
    kill_mutually = 6  # mutual annihilation of attacking and defending piece
