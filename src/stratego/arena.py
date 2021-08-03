from typing import Dict

from .core import Status, Team
from .utils import RollingMeter
from .game import Game

import os
import sys
from colorama import Fore, Style

from tqdm import tqdm
from timeit import default_timer as timer

import colorama

colorama.init()

red = Fore.RED
blue = Fore.BLUE
rs = Style.RESET_ALL


class Stats:
    def __init__(self):
        self.meter = {Team(0): RollingMeter(), Team(1): RollingMeter()}
        self.move_count = RollingMeter()
        self.time_count = RollingMeter()
        self.draws = 0

    def add_win(self, team: Team, move_count, time_count):
        self.meter[team].push(1)
        self.move_count.push(move_count)
        self.time_count.push(time_count)

    def add_draw(self, move_count, time_count):
        self.draws += 1
        self.move_count.push(move_count)
        self.time_count.push(time_count)

    def get_count_stats(self):
        meter_0 = self.meter[Team(0)]
        meter_1 = self.meter[Team(1)]
        return (
            meter_0.count,
            meter_0.avg,
            meter_1.count,
            meter_1.avg,
            self.move_count.count,
            self.move_count.avg,
            self.time_count.count,
            self.time_count.avg,
        )


def fight(
    game_env: Game,
    n_fights: int,
    show_game: bool = False,
    save_results: bool = False,
    folder: str = ".",
    filename: str = "",
):
    """
    Simulate num_sims many games of the agent of type agent_type_0 against the agent of
    type agent_type_1. If setup_0 or setup_1 are provided respectively, then take the pieces
    setup from those. If show_game is True, the game will be printed by the internal function.

    :return: None, writes results to a file named after the agents acting

    Parameters
    ----------
    save_results: bool (optional),
        whether to save the results to file.
        Requires :attr:`folder` and :attr:`filename`, if True. Defaults to False.
    folder: str (optional)
        the folder in which to save the stats. Only needed if :attr:`save_results` is True.
    filename: str (optional),
        the filename to which the stats are saved.
    n_fights: integer,
        number of games to simulate
    show_game: (optional) boolean,
        whether to plot the game while it is played or not
    """
    stats = Stats()
    ag_types = dict()
    for agent in game_env.agents:
        ag_types[agent.team] = type(agent).__name__

    # simulate games
    for sim in (bar := tqdm(range(n_fights), colour="green", file=sys.stdout)) :
        # reset setup with new setup if none given
        game_env.reset()

        time_s = timer()

        game_status = game_env.run_game(show=show_game)

        elapsed_t = timer() - time_s

        if game_status == Status.win_blue:
            stats.add_win(Team.blue, game_env.state.turn_counter, elapsed_t)
        elif game_status == Status.win_red:
            stats.add_win(Team.red, game_env.state.turn_counter, elapsed_t)
        else:
            stats.add_draw(game_env.state.turn_counter, elapsed_t)

        if sim % 10 == 0:
            ag_0_res = f"{blue}BLUE{rs} ({blue}{ag_types[Team.blue]}{rs})".center(30)
            ag_1_res = f"{red}RED{rs}  ({red}{ag_types[Team.red]}{rs})".center(30)
            red_won = str(stats.meter[Team.red]).ljust(4)
            blue_won = str(stats.meter[Team.blue]).rjust(4)
            bar.set_description(
                f'{f"Game {sim + 1}/{n_fights}".center(10)} {ag_0_res} --> {blue_won} : {red_won} <-- {ag_1_res}'
                f"\t Draws: {stats.draws}"
            )

    if save_results:
        write_results(
            n_fights,
            folder,
            filename,
            ag_types,
            stats,
        )

    return stats


def write_results(n_fights: int, folder: str, filename: str, ag_types: Dict[Team, str], stats: Stats):
    if os.path.exists(os.path.join(folder, filename, ".txt")):
        suffix = 1
        while True:
            if os.path.exists(os.path.join(folder, filename, f"({suffix})", ".txt")):
                suffix += 1
            else:
                filename += f"({suffix})"
                break

    with open(os.path.join(folder, filename, ".txt"), "w") as file:
        file.write(
            "Statistics of {} vs. {} with {} games played.\n".format(
                ag_types[Team.blue], ag_types[Team.red], n_fights
            )
        )
        file.write(
            f"Overall computational time of simulation: {sum(stats.time_count.sum)} seconds.\n"
        )

        file.write(
            "\nAgent {} won {}/{} games (~{:.2f}%).\n".format(
                ag_types[Team.blue],
                stats.meter[Team.blue].sum,
                n_fights,
                100 * stats.meter[Team.blue].sum / n_fights,
            )
        )

        file.write(
            "\nAgent {} won {}/{} games (~{:.2f}%).\n".format(
                ag_types[Team.red],
                stats.meter[Team.red].sum,
                n_fights,
                100 * stats.meter[Team.red].sum / n_fights,
            )
        )

        file.write(
            f"\nAverage game duration overall: {stats.time_count.avg / n_fights:.2f} rounds\n"
        )
        file.write(f"Maximum number of rounds played: {stats.move_count.max} rounds\n")
        file.write(f"Minimum number of rounds played: {stats.move_count.min} rounds\n")
