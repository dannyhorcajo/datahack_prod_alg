from typing import List, Tuple
import math
import numpy as np


class Navigation:
    """Class for short-term path planning."""

    SENSORS = [(0.1067, 0.1382, 1),
               (0.1557, 0.1250, 0.8),
               (0.1909, 0.0831, 0.3),
               (0.2095, 0.0273, 0.1),
               (0.2095, -0.0273, -0.1),
               (0.1909, -0.0785, -0.3),
               (0.1558, -0.1203, -0.8),
               (0.1067, -0.1382, -1),
               (-0.1100, -0.1382, -1),
               (-0.1593, -0.1203, -2.2689),
               (-0.1943, -0.0785, -2.6180),
               (-0.2129, -0.0273, -2.9671),
               (-0.2129, 0.0273, 2.9671),
               (-0.1943, 0.0785, 2.6180),
               (-0.1593, 0.1203, 2.2689),
               (-0.1100, 0.1382, 1)]


    def __init__(self, dt: float):
        """Navigation class initializer.

        Args:
            dt: Sampling period [s].
            
        """
        self._dt = dt
        self._w_max = 2.5   # [rad/s]
        self._v_max = 0.5   # [m/s]
        self._a_max = 0.8     # [m/s^2]
        self._alpha_max = 1000     # [rad/s^2]

        self._KP = 20
        self._KD = 5
        self._K_angle = 0.05
        self._K_center = 0.45
        self._distance_wall_left = 0.35
        self._distance_wall_right = 0.45
        self._critical_distance = 0.15
        self._error_prior = 0
        self._bias = 0

        self._KP_path = 3
        self._KD_path = 0.5
        self._error_prior_path = 0

        self._operation_mode = 'SPIN'
        self._flag_turning = False        


    def explore(self, z_us: List[float], z_v: float, z_w: float, steps=1) -> Tuple[float, float]:
        """Wall following exploration algorithm.

        Args:
            z_us: Distance from every ultrasonic sensor to the closest obstacle [m].
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        v = 0
        w = 0

        # Reset operation mode
        self._operation_mode = 'SPIN'

        # Check if robot is at a dead end
        if not self._flag_turning:
            for i, distance in enumerate(z_us):
                if distance > 0.8 and i < 6 and i > 1:
                    self._operation_mode = 'CORRIDOR'

        if self._operation_mode == 'CORRIDOR':

            angle = 0
            
            # Detect direction of open space
            for i, distance in enumerate(z_us):
                if (distance == float('inf')) and i<8 and i>4:
                    angle += self.SENSORS[i][2]

            # print('angle:', angle)
            
            center = 0

            # If a side wall is missing and nothing up front, follow one wall
            # Wall at the right side
            if z_us[1] > 0.5 and z_us[6] != float('inf'):
                center = self._distance_wall_right - z_us[6]
                
            # Wall at the left side
            elif z_us[6] > 0.5 and z_us[1] != float('inf') and z_us[7] != float('inf'):
                # print('Follow wall left')
                center = z_us[1] - self._distance_wall_left

            # If walls at both sides, center car
            elif z_us[1] < 0.9 and z_us[6] < 0.9:
                center = z_us[1] - z_us[6]

            # PD control
            error = angle * self._K_angle + center * self._K_center
            derivative = (error - self._error_prior) / self._dt
            w = self._KP*error + self._KD*derivative + self._bias
            self._error_prior = error
            
            # Whenever there is no need to turn, increase speed to 1
            if (angle > -0.6) and (angle < 0.6) and (z_us[3] == float('inf')) and (z_us[4] == float('inf')):
                v = 0.9
            else:
                v = 0.6
            
        elif self._operation_mode == 'SPIN':
            self._flag_turning = True
            v = 0
            angle = 0
            for i, distance in enumerate(z_us):
                if (distance == float('inf')) and (i < 8):
                    angle += self.SENSORS[i][2]
            if angle>0: w = 1.5
            else: w = -1.5
            if (z_us[3] > 0.7) or (z_us[4] > 0.7):
                self._flag_turning = False
        
        # Print logs
        if steps % 5 == 0: 
            print('Sensores' + str(z_us[:]))
            print('v: ' + str(v))
            print('w: ' + str(w))

        # Take into account v_max, w_max and a_max, alpha_max
        if w > self._w_max:
            w = self._w_max
        if w < -self._w_max:
            w = -self._w_max

        a = (v - z_v)/self._dt
        if abs(a) > self._a_max:
            v = z_v + np.sign(a) * self._a_max * self._dt
        v = max(0, min(v, self._v_max))

        return v, w

    def path_follower(self, current_pos: Tuple[float, float, float], goal: Tuple[float, float], next_goal: Tuple[float, float], z_us: List[float]):
        """Path follower algorithm that calculates the commands to give to the robot in order to go from its current position to the goal.

        Args:
            current_pos: (x, y, theta) [m, m, rad]
            goal: (x, y) [m]
            next_goal: second point in the path (x, y) [m]
            z_us: Distance from every ultrasonic sensor to the closest obstacle [m].

        Returns:
            v: Linear velocity
            w: Angular velocity

        """
        # Angle deviation between the robot and the correct path
        beta = math.atan2((goal[1] - current_pos[1]), (goal[0] - current_pos[0]))
        if beta < 0: beta += 2*np.pi
        
        relative_angle = beta - current_pos[2]
        if relative_angle > np.pi : relative_angle = relative_angle - 2*np.pi
        if relative_angle < -np.pi : relative_angle = relative_angle + 2*np.pi

        # Angle deviation between the robot and the next goal
        gamma = math.atan2((next_goal[1] - current_pos[1]), (next_goal[0] - current_pos[0]))
        if gamma < 0: gamma += 2*np.pi

        relative_gamma = gamma - current_pos[2]
        if relative_gamma > np.pi : relative_gamma -= 2*np.pi
        if relative_gamma < -np.pi : relative_gamma += 2*np.pi

        center = 0
        if z_us[0] < self._critical_distance:
            center = z_us[0] - self._critical_distance
        elif z_us[7] < self._critical_distance:
            center = self._critical_distance - z_us[7]

        error = relative_angle + center
        derivative = (error - self._error_prior_path) / self._dt
        w = self._KP_path*error + self._KD_path*derivative + self._bias
        self._error_prior_path = error

        if w > self._w_max:
            w = self._w_max
        if w < -self._w_max:
            w = -self._w_max

        if abs(relative_gamma) < 0.35: # 20 grados
            v = 0.5
        elif abs(relative_gamma) < 1.05: # 60 grados
            v = 0.3
        else:
            v = 0.1

        return v, w
