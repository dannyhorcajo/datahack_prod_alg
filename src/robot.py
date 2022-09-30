from abc import ABC, abstractmethod


class Robot(ABC):
    """Abstract base class to control mobile robots."""

    def __init__(self, client_id: int, track: float, wheel_radius: float):
        """Robot class initializer.

        Args:
            client_id: CoppeliaSim connection handle.
            track: Distance between the centerline of two wheels on the same axle [m].
            wheel_radius: Radius of the wheels [m].

        """
        self._client_id = client_id
        self._track = track
        self._wheel_radius = wheel_radius
        self.position = (0, 0, 0)

    @abstractmethod
    def move(self, v: float, w: float):
        """Solve inverse kinematics and send commands to the motors.

        Args:
            v: Linear velocity of the robot center [m/s].
            w: Angular velocity of the robot center [rad/s].

        """
<<<<<<< HEAD
        new_position = tuple(np.random.randint(-100, 100, 2))
        self.position = new_position
=======
        # TODO Implement angular velocity
        return self.position[0] + v*0.1
>>>>>>> development

    @abstractmethod
    def sense(self):
        """Acquire sensor readings."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the robot."""
        pass

    @abstractmethod
    def get_position(self):
        """Get the robot position.

        Returns:
            The robot position.

        """
        return self.position

    @abstractmethod
    def set_position(self, position: tuple):
        """Set the robot position.

        Args:
            position: The robot position.

        """
        self.position = position

    @abstractmethod
    def beep():
        """Play a beep sound."""
        print("Beep!")

    @abstractmethod
    def dance():
        """Play a dance music."""
        print("Dance!")

    @abstractmethod
    def sing(song=None):
        """Play a singing music."""
        if song:
            print(f"Singing {song} (8)")
        else:
            print("Hello darkness my old friend...")
