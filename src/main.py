import math
import os
import sim
import time
import numpy as np

from particle_filter import ParticleFilter
from robot_p3dx import RobotP3DX
from navigation import Navigation
from typing import Tuple
from map import Map
from planning import Planning


def create_robot(client_id: int, x: float, y: float, theta: float):
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path, 'p3dx.ttm')

    rc, out_ints, _, _, _ = sim.simxCallScriptFunction(client_id, 'Maze', sim.sim_scripttype_childscript, 'createRobot', [], [x, y, theta], [model_path], "", sim.simx_opmode_blocking)
    robot_handle = out_ints[0]

    return rc, robot_handle


def goal_reached(robot_handle: int, goal: Tuple[float, float], localized: bool, tolerance: float = 0.1) -> bool:
    distance = float('inf')

    if localized:
        _, position = sim.simxGetObjectPosition(client_id, robot_handle, -1, sim.simx_opmode_buffer)
        distance = math.sqrt((position[0] - goal[0]) ** 2 + (position[1] - goal[1]) ** 2)

    return distance < tolerance


def point_reached(current_pos: Tuple[float, float, float], goal: Tuple[float, float], tolerance: float = 0.1) -> bool:

    distance = math.sqrt((current_pos[0] - goal[0]) ** 2 + (current_pos[1] - goal[1]) ** 2)

    return distance < tolerance


if __name__ == '__main__':
    # Connect to CoppeliaSim
    sim.simxFinish(-1)  # Close all pending connections
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 2000, 5)

    if client_id == -1:
        raise ConnectionError('Could not connect to CoppeliaSim. Make sure the application is open.')

    # Start simulation
    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_blocking)

    # Initial and final locations
    start = (2, -3, math.pi/2)
    goal = (0, 4)

    # Create the robot
    _, robot_handle = create_robot(client_id, start[0], start[1], start[2])
    sim.simxGetObjectPosition(client_id, robot_handle, -1, sim.simx_opmode_streaming)  # Initialize real position streaming

    # Execute a simulation step to get initial sensor readings
    sim.simxSynchronousTrigger(client_id)
    sim.simxGetPingTime(client_id)  # Make sure the simulation step has finished

    # Initialization
    dt = 0.05
    steps = 0
    robot = RobotP3DX(client_id, dt)
    navigation = Navigation(dt)
    localized = False
    path_calculated = False
    start_time = time.time()
    m = Map('map_project.json', sensor_range=RobotP3DX.SENSOR_RANGE, compiled_intersect=True, use_regions=True)
    pf = ParticleFilter(m, RobotP3DX.SENSORS, RobotP3DX.SENSOR_RANGE, particle_count=7000, v_noise=0.15, w_noise=0.15, sense_noise=0.55)
    action_costs = (2.0, 1.0, 2.0, 5.0)
    sensor_list = range(16)
    planning = Planning(m, action_costs)

    time_resample = 0
    time_plot_sense = 0
    resampling_rate = 30

    try:
        while not goal_reached(robot_handle, goal, localized):

            start = time.time()
            z_us, z_v, z_w = robot.sense()
            time_sense = time.time() - start

            # Resampling frequency
            time_resample = 0
            if steps % resampling_rate == 0 and steps != 0:
                if steps < 15:
                    num_particles = 5000
                    ratio_random = 0.1
                elif steps < 45:
                    num_particles = 4500
                    ratio_random = 0.05
                elif num_clusters > 5 and num_particles >= 2500:
                    num_particles = 2500
                    ratio_random = 0
                elif num_clusters > 4 and num_particles >= 1500:
                    num_particles = 1500
                    ratio_random = 0
                    sensor_list = [0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15]
                    resampling_rate = 25
                elif num_clusters >= 3 and num_particles >= 600:
                    num_particles = 600
                    ratio_random = 0
                    sensor_list = [0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15]
                    resampling_rate = 20
                elif num_clusters >= 2 and num_particles >= 200:
                    num_particles = 200
                    ratio_random = 0
                    sensor_list = [0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15]
                    resampling_rate = 10
                elif num_clusters >= 1 and num_particles >= 100:
                    num_particles = 100
                    ratio_random = 0
                    sensor_list = [0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15]
                    resampling_rate = 10
                else:
                    num_particles = 3250
                    ratio_random = 1
                    sensor_list = range(16)

                start = time.time()
                pf.resample(z_us, number_particles=num_particles, ratio_random=ratio_random, sensor_list=sensor_list)
                time_resample = time.time() - start

                localized, num_clusters, curr_pos = pf.get_current_position()
                print("Num clusters:", num_clusters)
                print("Num particles:", num_particles)
                if localized and not path_calculated:
                    path = planning.a_star(start=curr_pos, goal=goal, naive_search=False)
                    smoothed_path = planning.smooth_path(path, data_weight=0.1, smooth_weight=0.025)
                    if len(smoothed_path) != 1:
                        smoothed_path.pop(0)
                    planning.show(path, smoothed_path)
                    path_calculated = True

            time_plot_sense = 0
            # if steps%10==0:
            #     start = time.time()
            #     pf.show('Sense')
            #     time_plot_sense = time.time() - start

            if not localized:
                start = time.time()
                v, w = navigation.explore(z_us, z_v, z_w)
                time_navigation = time.time() - start
            else:
                _, _, curr_pos = pf.get_current_position()
                if point_reached(curr_pos, smoothed_path[0], tolerance=0.3) and len(smoothed_path) != 1:
                    smoothed_path.pop(0)
                start = time.time()
                if len(smoothed_path) == 1:
                    v, w = navigation.path_follower(curr_pos, smoothed_path[0], smoothed_path[0], z_us)
                else:
                    v, w = navigation.path_follower(curr_pos, smoothed_path[0], smoothed_path[1], z_us)
                time_navigation = time.time() - start

            start = time.time()
            pf.move(z_v, z_w, dt)
            time_move_pf = time.time() - start

            start = time.time()
            robot.move(v, w)
            time_move_robot = time.time() - start

            time_plot_move = 0
            if steps % 5 == 0:
                start = time.time()
                pf.show('Move')
                time_plot_move = time.time() - start

            print('Time_sense: {0:6.3f} s | Time_resample: {1:6.3f} s | Time_plot_sense: {2:6.3f} s   |   Time_navigation: {3:6.3f} s | Time_move_pf: {4:6.3f} s | Time_move_robot: {5:6.3f} s | time_plot_move: {6:6.3f}'.format(time_sense, time_resample, time_plot_sense, time_navigation, time_move_pf, time_move_robot, time_plot_move))

            # Execute the next simulation step
            sim.simxSynchronousTrigger(client_id)
            sim.simxGetPingTime(client_id)  # Make sure the simulation step has finished
            steps += 1

    except KeyboardInterrupt:  # Press Ctrl+C to break the infinite loop and gracefully stop the simulation
        pass

    # Display time statistics
    execution_time = time.time() - start_time
    print('\n')
    print('Simulated steps: {0:d}'.format(steps))
    print('Simulated time:  {0:.3f} s'.format(steps * dt))
    print('Execution time:  {0:.3f} s ({1:.3f} s/step)'.format(execution_time, execution_time / steps))
    print('')

    # Stop the simulation and close the connection
    sim.simxStopSimulation(client_id, sim.simx_opmode_blocking)
    sim.simxGetPingTime(client_id)  # Make sure the stop simulation command had time to arrive
    sim.simxFinish(client_id)
