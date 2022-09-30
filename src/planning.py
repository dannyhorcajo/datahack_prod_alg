import math
import numpy as np
import os

from map import Map
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple


class Planning:
    """Class to plan the optimal path to a given location."""

    def __init__(self, map_object: Map, action_costs: Tuple[float, float, float, float]):
        """Planning class initializer.

        Args:
            map_object: Map of the environment.
            action_costs: Cost of of moving one cell left, right, up, and down.

        """
        self._map = map_object

        self._actions = np.array([
            (-1, 0),  # Move one cell left
            (0, 1),   # Move one cell up
            (1, 0),   # Move one cell right
            (0, -1)   # Move one cell down
        ])

        self._action_costs = action_costs

    def a_star(self, start: Tuple[float, float, float], goal: Tuple[float, float], naive_search=False) -> List[Tuple[float, float]]:
        """Computes the optimal path to a given goal location using the A* algorithm.

        Args:
            start: Initial location in (x, y) format.
            goal: Destination in (x, y) format.

        Returns:
            Path to the destination. The first value corresponds to the initial location.

        """
        # Check both points are inside the map and raise exception otherwise
        start_rc = start_row, start_column = self._xy_to_rc(start)
        goal_rc = goal_row, goal_column = self._xy_to_rc(goal)
        orientation = (0, 0)
        angle = start[2]/(2*np.pi)
        if angle <= 0.125 or angle > 0.875:
            orientation = (1, 0)
        elif angle <= 0.375:
            orientation = (0, 1)
        elif angle <= 0.625:
            orientation = (-1, 0)
        else:
            orientation = (0, -1)

        if self._map.grid_map[start_rc] != 1 and self._map.grid_map[goal_rc] != 1:
            heuristic = self._compute_heuristic(goal, naive_search)
            heuristic_map = heuristic.copy()
            open_list = {(start_row, start_column): (0 + heuristic[start_row, start_column], 0)}  # Format: {(r, c): (f, g)}
            closed_list = set()  # Format: (r, c)
            ancestors = {}
            num_steps = 0

            while open_list:
                num_steps += 1
                current_node = r, c = min(open_list, key=lambda k: open_list.get(k)[0])
                if num_steps != 1:
                    ancestor_current_node = self._xy_to_rc(ancestors[self._rc_to_xy(current_node)])
                    orientation = (current_node[1] - ancestor_current_node[1], ancestor_current_node[0] - current_node[0])

                _, g = open_list.pop(current_node)

                if current_node == goal_rc:
                    print('Goal reached!')
                    print('Num steps:', num_steps)
                    return self._reconstruct_path(start=self._rc_to_xy(start_rc), goal=self._rc_to_xy(goal_rc), ancestors=ancestors)

                for action, action_cost in zip(self._actions, self._rotate_action_cost(orientation)):
                    action_rc = self._action_xy_to_rc(action)
                    neighbor = r + action_rc[0], c + action_rc[1]
                    if self._map.contains(self._rc_to_xy(neighbor)):
                        if (neighbor not in open_list) and (neighbor not in closed_list):
                            # coste de orientar
                            g_new = g + action_cost
                            f_new = g_new + heuristic[neighbor]
                            open_list[neighbor] = (f_new, g_new)
                            heuristic_map[neighbor] = f_new
                            ancestors[self._rc_to_xy(neighbor)] = self._rc_to_xy(current_node)

                closed_list.add(current_node)
        else:
            raise ValueError('Start and/or goal coordinates outside of map')

    @staticmethod
    def smooth_path(path, data_weight: float = 0.1, smooth_weight: float = 0.1, tolerance: float = 1e-6) -> \
            List[Tuple[float, float]]:
        """Computes a smooth trajectory from a Manhattan-like path.

        Args:
            path: Non-smoothed path to the goal (start location first).
            data_weight: The larger, the more similar the output will be to the original path.
            smooth_weight: The larger, the smoother the output path will be.
            tolerance: The algorithm will stop when after an iteration the smoothed path changes less than this value.

        Returns: Smoothed path (initial location first) in (x, y) format.

        """

        smooth = path.copy()
        change = tolerance

        while change >= tolerance:
            change = 0
            for i in range(len(smooth)):
                if i not in [0, len(smooth)-1]:
                    smooth_prev = smooth[i]
                    smooth_x = smooth[i][0] + data_weight * (path[i][0] - smooth[i][0]) + smooth_weight * (smooth[i+1][0] + smooth[i-1][0] - 2*smooth[i][0])
                    smooth_y = smooth[i][1] + data_weight * (path[i][1] - smooth[i][1]) + smooth_weight * (smooth[i+1][1] + smooth[i-1][1] - 2*smooth[i][1])
                    smooth[i] = smooth_x, smooth_y
                    change += abs(smooth[i][0] - smooth_prev[0]) + abs(smooth[i][1] - smooth_prev[1])

        return smooth

    @staticmethod
    def plot(axes, path: List[Tuple[float, float]], smoothed_path: List[Tuple[float, float]] = ()):
        """Draws a path.

        Args:
            axes: Figure axes.
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).

        Returns:
            axes: Modified axes.

        """
        x_val = [x[0] for x in path]
        y_val = [x[1] for x in path]

        axes.plot(x_val, y_val)  # Plot the path
        axes.plot(x_val[1:-1], y_val[1:-1], 'bo', markersize=4)  # Draw blue circles in every intermediate cell

        if smoothed_path:
            x_val = [x[0] for x in smoothed_path]
            y_val = [x[1] for x in smoothed_path]

            axes.plot(x_val, y_val, 'y')  # Plot the path
            axes.plot(x_val[1:-1], y_val[1:-1], 'yo', markersize=4)  # Draw yellow circles in every intermediate cell

        axes.plot(x_val[0], y_val[0], 'rs', markersize=7)  # Draw a red square at the start location
        axes.plot(x_val[-1], y_val[-1], 'g*', markersize=12)  # Draw a green star at the goal location

        return axes

    def show(self, path, smoothed_path=(), figure_number: int = 1, title: str = 'Path', block: bool = False,
             figure_size: Tuple[float, float] = (7, 7), save_figure: bool = False, save_dir: str = 'img'):
        """Displays a given path on the map.

        Args:
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).
            figure_number: Any existing figure with the same value will be overwritten.
            title: Plot title.
            blocking: True to stop program execution until the figure window is closed.
            figure_size: Figure window dimensions.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure, axes = plt.subplots(1, 1, figsize=figure_size, num=figure_number)
        axes = self._map.plot(axes)
        axes = self.plot(axes, path, smoothed_path)
        axes.set_title(title)
        figure.tight_layout()  # Reduce white margins

        plt.show(block=block)
        plt.pause(0.0001)  # Wait for 0.1 ms or the figure won't be displayed

        if save_figure:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            file_name = str(title.lower() + '.png')
            file_path = os.path.join(save_dir, file_name)
            figure.savefig(file_path)

    def _compute_heuristic(self, goal: Tuple[float, float], naive_search: bool = False, increase_factor: float = 0.5) -> np.ndarray:
        """Creates an admissible heuristic with a corresponding increase factor.

        Args:
            goal: Destination location in (x,y) coordinates.
            naive_search: compute with naive search.
            increase_factor: definition of the increase factor.

        Returns:
            Admissible heuristic.

        """
        goal_r, goal_c = self._xy_to_rc(goal)
        rows, columns = np.shape(self._map.grid_map)

        heuristic_map = np.zeros((rows, columns))

        if not naive_search:
            for index, _ in np.ndenumerate(heuristic_map):
                row, col = index
                if not self._map.contains(self._rc_to_xy((row, col))):
                    heuristic_map[index] = 1000
                else:
                    heuristic_map[index] = (abs(goal_r - row) + abs(goal_c - col)) * increase_factor

        return heuristic_map

    def _reconstruct_path(self, start: Tuple[float, float], goal: Tuple[float, float],
                          ancestors: Dict[Tuple[int, int], Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Computes the trajectory from the start to the goal location given the ancestors of a search algorithm.

        Args:
            start: Initial location in (x, y) format.
            goal: Goal location in (x, y) format.
            ancestors: Matrix that contains for every cell, None or the (x, y) ancestor from which it was opened.

        Returns: Path to the goal (start location first) in (x, y) format.

        """

        path = []
        current_node = goal

        while current_node != start:
            path.append(current_node)
            current_node = ancestors[current_node]

        path.append(start)
        path.reverse()

        return path

    def _xy_to_rc(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """Converts (x, y) coordinates of a metric map to (row, col) coordinates of a grid map.

        Args:
            xy: (x, y) [m].

        Returns:
            rc: (row, col) starting from (0, 0) at the top left corner.

        """
        map_rows, map_cols = np.shape(self._map.grid_map)

        x = round(xy[0])
        y = round(xy[1])

        row = int(map_rows - (y + math.ceil(map_rows / 2.0)))
        col = int(x + math.floor(map_cols / 2.0))

        return row, col

    def _rc_to_xy(self, rc: Tuple[int, int]) -> Tuple[float, float]:
        """Converts (row, col) coordinates of a grid map to (x, y) coordinates of a metric map.

        Args:
            rc: (row, col) starting from (0, 0) at the top left corner.

        Returns:
            xy: (x, y) [m].

        """
        map_rows, map_cols = np.shape(self._map.grid_map)
        row, col = rc

        x = col - math.floor(map_cols / 2.0)
        y = map_rows - (row + math.ceil(map_rows / 2.0))

        return x, y

    def _action_xy_to_rc(self, xy: Tuple[int, int]) -> Tuple[float, float]:
        """Converts (x, y) actions of a metric map to (row, col) actions of a grid map.

        Args:
            xy: Actions (x, y) [m].

        Returns:
            rc: Actions (row, col) starting from (0, 0) at the top left corner.

        """
        rc = -xy[1], xy[0]

        return rc

    def _rotate_action_cost(self, orientation: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """Rotates the action cost tuple regarding the robot orientation.

        Args:
            orientation (Tuple[int, int]): Actual orientation of the robot with respect to the map

        Returns:
            Tuple[float, float, float, float]: Action costs
        """
        action_costs = self._action_costs       # Looking up
        if orientation == (1, 0):               # Looking left
            action_costs = np.roll(self._action_costs, 1)
        elif orientation == (0, -1):            # Looking down
            action_costs = np.roll(self._action_costs, 2)
        elif orientation == (-1, 0):            # Looking left
            action_costs = np.roll(self._action_costs, -1)

        return action_costs


def test():
    """Function used to test the Planning class independently."""
    m = Map('map_project.json', sensor_range=1.0, compiled_intersect=False, use_regions=False)

    start = (-4.0, -4.0, 5*np.pi/6)
    goal = (4.0, 4.0)
    action_costs = (1.5, 1.0, 1.5, 5.0)  # Left up right down / It asumes robot looks up
    naive_search = False

    planning = Planning(m, action_costs)
    path = planning.a_star(start, goal, naive_search)
    smoothed_path = planning.smooth_path(path, data_weight=0.1, smooth_weight=0.1)
    planning.show(path, smoothed_path, block=True)


if __name__ == '__main__':
    test()
