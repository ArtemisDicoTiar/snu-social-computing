import copy
import json
import random
import shutil
from functools import reduce
from itertools import product
from math import e
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class LandMap:
    def __init__(
            self,
            width: int, height: int,
            races_ratio: List[float] = None,
            empty_houses: int = 7000,
            sim_threshold: float = 0.8,
            n_iter: int = 100,
    ) -> None:
        self.save_path = Path(
            f"./images/w{width}-h{height}-rr{races_ratio}-eh{empty_houses}-st{sim_threshold}-mni{n_iter}"
        )
        if self.save_path.exists():
            shutil.rmtree(self.save_path)
        self.save_path.mkdir()

        self.width = width
        self.height = height
        self.races_ratio = races_ratio

        self.sim_threshold = sim_threshold
        self.n_iter = n_iter

        self.n_empty = empty_houses

        self.n_races = [
            int((self.width * self.height - self.n_empty) * race_ratio)
            for race_ratio in races_ratio
        ]

        self.agents = [[typ + 1] for typ, _ in enumerate(self.n_races)]

        self.houses = [0] * self.n_empty \
                      + reduce(lambda i, j: i + j, [[typ + 1] * n_race for typ, n_race in enumerate(self.n_races)])
        random.shuffle(self.houses)
        self.houses = np.array(self.houses).reshape((self.width, self.height))

        # {0: 'b', 1: 'r', 2: 'g', 3: 'c', 4: 'm', 5: 'y', 6: 'k'}
        colors = [(1, 1, 1), (0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 0)]
        self.cmap = LinearSegmentedColormap.from_list("tmp", colors[:len(self.races_ratio) + 1])

        self.n_changes = []

    def _check_valid_coordinate(self, x: int, y: int):
        x_condition = 0 < x < (self.width - 1)
        y_condition = 0 < y < (self.height - 1)
        return x_condition and y_condition

    def is_unsatisfied(self, x: int, y: int) -> bool:
        race = self.houses[x, y]
        cnt_sim = 0
        cnt_diff = 0

        surroundings = [
            (x, y - 1),  # north
            (x + 1, y - 1),  # north_east
            (x + 1, y),  # east
            (x + 1, y + 1),  # south_east
            (x, y + 1),  # south
            (x - 1, y + 1),  # south_west
            (x - 1, y),  # west
            (x - 1, y - 1),  # north_west
        ]

        for surrounding in surroundings:
            x, y = surrounding

            if not self._check_valid_coordinate(x, y):
                # this coordinate is invalid.
                continue
            if self.houses[x, y] == 0:
                # this coordinate is empty.
                continue

            if self.houses[x, y] == race:
                cnt_sim += 1
            else:
                cnt_diff += 1

        if cnt_sim + cnt_diff > 0:
            return cnt_sim / (cnt_sim + cnt_diff) < self.sim_threshold

        return False

    def move_to_empty(self, x: int, y: int):
        agent_race = self.houses[x, y]
        empty_houses = (self.houses == 0).nonzero()
        target_house = random.choice(list(zip(*empty_houses)))
        self.houses[target_house] = agent_race
        self.houses[x, y] = 0

    def plot(self, n_iter: int):
        fig, ax = plt.subplots()

        ax.imshow(self.houses, cmap=self.cmap)
        ax.set_title(f"Iteration {n_iter}", fontsize=10, fontweight="bold")
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(self.save_path / f"ni{n_iter}.png")
        plt.close(fig)

    def loop(self):
        for i in range(self.n_iter):
            old_agents = list(zip(*(self.houses != 0).nonzero()))
            n_changes = 0
            for agent in old_agents:
                if self.is_unsatisfied(*agent):
                    self.move_to_empty(*agent)
                    n_changes += 1

            self.n_changes.append(n_changes)
            print(f"Iteration: {i}, # of changes: {n_changes}")

            if i % 10 == 0:
                self.plot(i)
            if n_changes == 0:
                print(f"STOP Iteration: {i}")
                break


def hyper_param_experiment():
    records = {}
    for n_empty_house in range(2500, 7500, 2500):
        for sim_threshold_ten in [0.3, 0.5, 0.8]:
            sim_thresh = sim_threshold_ten

            for race_ratio in [0.01, 0.1, 0.2, 0.5]:
                left_ratio = race_ratio
                race_ratios = [left_ratio, round(1 - left_ratio, 3)]
                print(f"===={race_ratios}===={n_empty_house}===={sim_thresh}")
                world = LandMap(150, 150,
                                races_ratio=race_ratios, empty_houses=n_empty_house, sim_threshold=sim_thresh)
                world.loop()
                records[f"eh{n_empty_house}_st{sim_thresh}_rr{race_ratio}"] = world.n_changes

    with open("./grid_search_eh_st_rr", "w") as fp:
        json.dump(records, fp)


def multi_race_hyper_param_experiment():
    records = {}
    for num_race in [4, 5]:
        for sim_threshold_ten in [0.3, 0.5]:
            sim_thresh = sim_threshold_ten

            race_ratio = [round(1 / num_race, 3)] * num_race
            print(f"===={race_ratio}===={2500}===={sim_thresh}")
            world = LandMap(150, 150,
                            races_ratio=race_ratio, empty_houses=2500, sim_threshold=sim_thresh)
            world.loop()
            records[f"eh{2500}_st{sim_thresh}_rr{race_ratio}"] = world.n_changes

    with open("./grid_search_multi_race", "w") as fp:
        json.dump(records, fp)


def experiment(race_ratios, n_empty_house, sim_thresh):
    print(f"===={race_ratios}===={n_empty_house}===={sim_thresh}")
    world = LandMap(150, 150,
                    races_ratio=race_ratios, empty_houses=n_empty_house, sim_threshold=sim_thresh)
    world.loop()



if __name__ == '__main__':
    multi_race_hyper_param_experiment()



