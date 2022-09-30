import numpy as np
import sim

from robot import Robot
from typing import Any, Dict, List, Tuple


class RobotP3DX(Robot):
    """Class to control the Pioneer 3-DX robot."""

    # Constants
    SENSOR_RANGE = 1.0     # Ultrasonic sensor range [m]
    TRACK = 0.33           # Distance between same axle wheels [m]
    WHEEL_RADIUS = 0.0975  # Radius of the wheels [m]

    # Sensor location and orientation (x, y, theta) in the robot coordinate frame
    SENSORS = [(0.1067, 0.1382, 1.5708),
               (0.1557, 0.1250, 0.8727),
               (0.1909, 0.0831, 0.5236),
               (0.2095, 0.0273, 0.1745),
               (0.2095, -0.0273, -0.1745),
               (0.1909, -0.0785, -0.5236),
               (0.1558, -0.1203, -0.8727),
               (0.1067, -0.1382, -1.5708),
               (-0.1100, -0.1382, -1.5708),
               (-0.1593, -0.1203, -2.2689),
               (-0.1943, -0.0785, -2.6180),
               (-0.2129, -0.0273, -2.9671),
               (-0.2129, 0.0273, 2.9671),
               (-0.1943, 0.0785, 2.6180),
               (-0.1593, 0.1203, 2.2689),
               (-0.1100, 0.1382, 1.5708)]

    def __init__(self, client_id: int, dt: float):
        """Pioneer 3-DX robot class initializer.

        Args:
            client_id: CoppeliaSim connection handle.
            dt: Sampling period [s].

        """
        Robot.__init__(self, client_id, track=self.TRACK, wheel_radius=self.WHEEL_RADIUS)
        self._dt = dt
        self._motors = self._init_motors()
        self._sensors = self._init_sensors()

    def move(self, v: float, w: float):
        """Solve inverse differential kinematics and send commands to the motors.

        Args:
            v: Linear velocity of the robot center [m/s].
            w: Angular velocity of the robot center [rad/s].

        """

        # Solve inverse differential kinematics
        w_left = (v - self.TRACK/2 * w) / self.WHEEL_RADIUS
        w_right = (v + self.TRACK/2 * w) / self.WHEEL_RADIUS

        # Set motor speeds
        rc_right = sim.simxSetJointTargetVelocity(self._client_id, self._motors['right'], w_right, sim.simx_opmode_oneshot)
        rc_left = sim.simxSetJointTargetVelocity(self._client_id, self._motors['left'], w_left, sim.simx_opmode_oneshot)

        if (rc_right != 0) or (rc_left != 0):
            # print("Error moving")
            # print('w_right: ' + str(w_right))
            # print('w_left: ' + str(w_left))
            pass

        pass

    def sense(self) -> Tuple[List[float], float, float]:
        """Read ultrasonic sensors and encoders.

        Returns:
            z_us: Distance from every ultrasonic sensor to the closest obstacle [m].
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        """
        # Read ultrasonic sensors
        z_us = [float('inf')] * len(self.SENSORS)
        distance = float('inf')

        for pos, sensor_handle in enumerate(self._sensors):
            rc, is_valid, detected_point, _, _ = sim.simxReadProximitySensor(self._client_id, sensor_handle, sim.simx_opmode_buffer)
            if is_valid:
                # Compute the Euclidean distance to the obstacle
                distance = np.linalg.norm(detected_point)
                z_us[pos] = distance

        # Read encoders
        z_v, z_w = self._sense_encoders()

        return z_us, z_v, z_w

    def _init_encoders(self):
        """Initialize encoder streaming."""
        sim.simxGetFloatSignal(self._client_id, 'leftEncoder', sim.simx_opmode_streaming)
        sim.simxGetFloatSignal(self._client_id, 'rightEncoder', sim.simx_opmode_streaming)

    def _init_motors(self) -> Dict[str, int]:
        """Acquire motor handles.

        Returns: {'left': handle, 'right': handle}

        """
        motors = {'left': None, 'right': None}
        motor_names = {'left': 'Pioneer_p3dx_leftMotor', 'right': 'Pioneer_p3dx_rightMotor'}

        for motor_name in list(motors.keys()):
            # Acquire handles (in _init_motors)
            rc, handle = sim.simxGetObjectHandle(self._client_id, motor_names[motor_name], sim.simx_opmode_blocking)

            motors[motor_name] = handle

        return motors

    def _init_sensors(self) -> List[Any]:
        """Acquire ultrasonic sensor handles and initialize US and encoder streaming.

        Returns: List with ultrasonic sensor handles.

        """
        self._init_encoders()

        sensors = [None] * len(self.SENSORS)

        sensor_name = 'Pioneer_p3dx_ultrasonicSensor'

        for sensor_pos in range(len(sensors)):
            sensor_full_name = sensor_name + str(sensor_pos + 1)
            rc, handle = sim.simxGetObjectHandle(self._client_id, sensor_full_name, sim.simx_opmode_blocking)

            sensors[sensor_pos] = handle
            sim.simxReadProximitySensor(self._client_id, handle, sim.simx_opmode_streaming)

        return sensors

    def _sense_encoders(self) -> Tuple[float, float]:
        """Solve forward differential kinematics from encoder readings.

        Returns:
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        """
        rc_right, radR = sim.simxGetFloatSignal(self._client_id, 'rightEncoder', sim.simx_opmode_buffer)
        rc_left, radL = sim.simxGetFloatSignal(self._client_id, 'leftEncoder', sim.simx_opmode_buffer)

        if (rc_right != 0) or (rc_left != 0):
            print("Error sensing encoders")

        # Forward kinematics
        omega_right = radR / self._dt
        omega_left = radL / self._dt

        z_v = (omega_right + omega_left) * self.WHEEL_RADIUS / 2
        z_w = (omega_right - omega_left) * self.WHEEL_RADIUS / self.TRACK

        return z_v, z_w
