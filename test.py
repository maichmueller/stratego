
import colorama
from colorama import Fore, Back, Style
import sys
import time

def main():
    for x in range(0, 1000):
        print(x, end=" ")
        time.sleep(0.5)


def print_round_results(i, n, ag_0, ag_1, red_won, blue_won):
    red = Fore.RED
    blue = Fore.BLUE
    rs = Style.RESET_ALL
    print(f'\r{f"Game {i}/{n}".center(10)} : {f"{red}{str(ag_0).center(10)}{rs}"}{red_won} vs '
          f'{blue_won} {f"{blue}{str(ag_1).center(10)}{rs}"}',
          end='', flush=True)


if __name__ == '__main__':
    main()
