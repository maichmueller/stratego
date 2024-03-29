from abc import abstractmethod, ABC
from typing import Optional, TYPE_CHECKING

# if TYPE_CHECKING:
from ..core import ActionMap
from ..agent import Agent, RLAgent, DRLAgent

from ..game import Game
from ..learning import RewardToken

from copy import deepcopy

from pickle import Pickler, Unpickler
import os
from multiprocessing import Lock


class RLAlgorithm(ABC):
    """
    Base Class for a Reinforcement Learning algorithm.
    """

    def __init__(
        self,
        game: Game,
        student: RLAgent,
        action_map: ActionMap,
    ):
        assert isinstance(
            student, RLAgent
        ), f"Student agent to coach has to be of type '{RLAgent}'. Given type '{type(self.student).__name__}'"
        self.student = student

        self.action_map = action_map
        self.game = game

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Train the reinforcement learning agent according to the method defined in the concrete class.
        """
        raise NotImplementedError

    @staticmethod
    def reward_agent(agent: Agent, reward: RewardToken):
        if isinstance(agent, RLAgent):
            agent.add_reward(reward)

    def _new_state(self, lock: Optional[Lock] = None):
        """
        Resets the game and returns a deepcopy of the new state in a thread-safe manner, if the lock is provided.
        """
        gen_state = lambda: deepcopy(self.game.reset().state)
        if lock is not None:
            lock.lock()
            state = gen_state()
            lock.unlock()
        else:
            state = gen_state()
        return state


class DRLAlgorithm(RLAlgorithm, ABC):
    """
    Base Class for a Deep Reinforcement Learning algorithm.
    """

    def __init__(
        self,
        game: Game,
        student: DRLAgent,
        action_map: ActionMap,
        model_folder: str = "./checkpoints/models",
        train_data_folder: str = "./checkpoints/data",
    ):
        super().__init__(game, student, action_map)
        self.model_folder = model_folder
        self.train_data_folder = train_data_folder
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(train_data_folder):
            os.makedirs(train_data_folder)

        assert isinstance(
            student, DRLAgent
        ), f"Student agent to coach has to be of type '{DRLAgent}'. Given type '{type(self.student).__name__}'"

        self.action_map = action_map
        self.game = game

    def load_train_data(self, filename: str):
        data_filepath = os.path.join(self.train_data_folder, filename, ".data")
        with open(data_filepath, "rb") as f:
            train_data = Unpickler(f).load()
        return train_data

    def save_train_data(self, data, filename: str):
        folder = self.train_data_folder
        filename = os.path.join(folder, filename, ".data")
        with open(filename, "wb+") as f:
            Pickler(f).dump(data)
