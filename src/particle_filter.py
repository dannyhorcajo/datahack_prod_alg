import math
import numpy as np
import os
import random

from map import Map
from matplotlib import pyplot as plt
from typing import List, Tuple
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from sklearn.cluster import DBSCAN


class ParticleFilter:
    """Particle filter implementation."""

    def __init__(self, map_object: Map, sensors: List[Tuple[float, float, float]],
                 sensor_range: float, particle_count: int = 200, sense_noise: float = 0.5,
                 v_noise: float = 0.05, w_noise: float = 0.05, figure_size: Tuple[float, float] = (7, 7)):
        """Particle filter class initializer.

        Args:
            map_object: Map of the environment.
            sensors: Robot sensors location [m] and orientation [rad] in the robot coordinate frame (x, y, theta).
            sensor_range: Sensor measurement range [m]
            particle_count: Number of particles.
            sense_noise: Measurement standard deviation [m].
            v_noise: Linear velocity standard deviation [m/s].
            w_noise: Angular velocity standard deviation [rad/s].
            figure_size: Figure window dimensions.

        """
        self._map = map_object
        self._sensors = sensors
        self._sense_noise = sense_noise
        self._sensor_range = sensor_range
        self._v_noise = v_noise
        self._w_noise = w_noise
        self._iteration = 0
        self._centroid = (0, 0, np.pi)

        self._particles = self._init_particles(particle_count)
        self._ds, self._phi = self._init_sensor_polar_coordinates(sensors)
        self._figure, self._axes = plt.subplots(1, 1, figsize=figure_size)

    def move(self, v: float, w: float, dt: float):
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].
            dt: Sampling time [s].

        """
        self._iteration += 1

        for i, particle in enumerate(self._particles):
            x, y, theta = particle
            v_noise = v + np.random.normal(scale=self._v_noise)
            w_noise = w + np.random.normal(scale=self._w_noise)

            new_x = v_noise * np.cos(theta) * dt + x
            new_y = v_noise * np.sin(theta) * dt + y
            new_theta = (theta + w_noise * dt) % (2 * np.pi)

            intersection, _ = self._map.check_collision([(x, y), (new_x, new_y)])

            if not intersection:
                self._particles[i] = (new_x, new_y, new_theta)
            else:
                self._particles[i] = (intersection[0], intersection[1], new_theta)
        pass

    def resample(self, measurements: List[float], number_particles: int = -1, ratio_random: float = 0, sensor_list: List[int] = range(16)):
        """Samples a new set of set of particles using the resampling wheel method.

        Args:
            measurements: Sensor measurements [m].
            number_particles: Number of particles to generate (reduce must be set to true).
            ratio_random: Percentage of particles to re-generate from scratch following a uniform distribution.
            sensor_list: Sensor list to compute the measurement probability.

        """
        # Get and normalize weights
        weights = []
        for part in self._particles:
            weights.append(self._measurement_probability(measurements, part, sensor_list))

        norm_weights = [weight/np.sum(weights) for weight in weights]

        # Resample using the wheel method
        if number_particles != -1: 
            n = number_particles
        else: 
            n = len(self._particles)

        n_random = int(n * ratio_random)
        n -= n_random
        
        index = np.random.randint(0, n-1)
        beta = 0
        w_max = np.max(norm_weights)
        new_particles = []

        if n > 0:
            for i in range(1, n+1):
                beta += random.uniform(0, 2*w_max)
                while norm_weights[index] < beta:
                    beta -= norm_weights[index]
                    index = (index + 1) % n
                new_particles.append(self._particles[index])

        if n_random > 0: new_particles += self._init_particles(n_random).tolist()

        self._particles = np.array(new_particles)

        pass

    def plot(self, axes, orientation: bool = True):
        """Draws particles.

        Args:
            axes: Figure axes.
            orientation: Draw particle orientation.

        Returns:
            axes: Modified axes.

        """
        if orientation:
            dx = [math.cos(particle[2]) for particle in self._particles]
            dy = [math.sin(particle[2]) for particle in self._particles]
            axes.quiver(self._particles[:, 0], self._particles[:, 1], dx, dy, color='b', scale=15, scale_units='inches')
            axes.plot(self._centroid[0], self._centroid[1], 'ro', markersize=4)
        else:
            axes.plot(self._particles[:, 0], self._particles[:, 1], 'bo', markersize=1)
            axes.plot(self._centroid[0], self._centroid[1], 'ro', markersize=4)

        return axes

    def show(self, title: str = '', orientation: bool = True, display: bool = True,
             save_figure: bool = False, save_dir: str = 'img'):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            orientation: Draw particle orientation.
            display: True to open a window to visualize the particle filter evolution in real-time. Time consuming.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure = self._figure
        axes = self._axes
        axes.clear()

        axes = self._map.plot(axes)
        axes = self.plot(axes, orientation)

        axes.set_title(title + ' (Iteration #' + str(self._iteration) + ')')
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=False)
            plt.pause(0.001)  # Wait for 0.1 ms or the figure won't be displayed

        if save_figure:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            file_name = str(self._iteration).zfill(4) + ' ' + title.lower() + '.jpg'
            file_path = os.path.join(save_dir, file_name)
            figure.savefig(file_path)

    def get_current_position(self, eps: float = 0.3, min_samples: int = 25):  # -> List[bool, Tuple[float, float, float]]:
        """Estimates the current position from the particles' pose using DBSCAN clustering.

        Args:
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
            min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.

        Returns:
            localized (bool): If a single position has been estimated
            n_clusters_ (int): Number of clusters found with the particles.
            current_pos (Tuple[float, float, float]): The estimated current position (x, y, theta) when n_clusters_ = 1.

        """
        localized = False
        current_pos = tuple()

        positions = self._particles[:, 0:2]

        # Compute DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(positions)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters_ == 1:
            localized = True

            poses = self._particles[:, -1]

            unique_labels = set(labels)
            unique_labels.discard(-1)   # Remove noise label

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True     # True where label != -1

            class_member_mask = (labels == 0)   # True where label == 0
            members = positions[class_member_mask & core_samples_mask]   # Actual points in the cluster
            centroid = np.average(members, axis=0)
            pose = np.median(poses)

            current_pos = (centroid[0], centroid[1], pose)
            self._centroid = current_pos

        return localized, n_clusters_, current_pos

    def _init_particles(self, particle_count: int) -> np.ndarray:
        """Draws N random valid particles.

        The particles are guaranteed to be inside the map and
        can only have the following orientations [0, pi/2, pi, 3pi/2].

        Args:
            particle_count: Number of particles.

        Returns: A numpy array of tuples (x, y, theta).

        """

        particles = np.zeros((particle_count, 3), dtype=object)
        num_generated_particles = 0
        thetas = [0, np.pi/2, np.pi, 3*np.pi/2]

        x_min, y_min, x_max, y_max = self._map.bounds()

        while num_generated_particles < particle_count:

            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            i_theta = np.random.randint(4)
            theta = thetas[int(i_theta)]

            if self._map.contains((x, y)):
                particles[num_generated_particles] = np.array([x, y, theta])
                num_generated_particles += 1

        return particles

    @staticmethod
    def _init_sensor_polar_coordinates(sensors: List[Tuple[float, float, float]]) -> Tuple[List[float], List[float]]:
        """Converts the robots sensor location and orientation to polar coordinates wrt to the robot's coordinate frame.

        Args:
            sensors: Robot sensors location [m] and orientation [rad] (x, y, theta).

        Return:
            ds: List of magnitudes [m].
            phi: List of angles [rad].

        """
        ds = [math.sqrt(sensor[0] ** 2 + sensor[1] ** 2) for sensor in sensors]
        phi = [math.atan2(sensor[1], sensor[0]) for sensor in sensors]

        return ds, phi

    def _sense(self, particle: Tuple[float, float, float], sensor_list: List[int] = range(16)) -> List[float]:
        """Obtains the predicted measurement of every sensor given the robot's location.

        Args:
            particle: Particle pose (x, y, theta) in [m] and [rad].
            sensor_list: Sensor list to compute the measurement probability.

        Returns: List of predicted measurements; inf if a sensor is out of range.

        """
        rays = self._sensor_rays(particle)

        z_hat = []

        for i, ray in enumerate(rays):
            if i in sensor_list:
                _, dist = self._map.check_collision(ray, compute_distance=True)
                z_hat.append(dist)

        return z_hat

    @staticmethod
    def _gaussian(mu: float, sigma: float, x: float) -> float:
        """Computes the value of a Gaussian.

        Args:
            mu: Mean.
            sigma: Standard deviation.
            x: Variable.

        Returns:
            float: Gaussian.

        """
        gaussian = math.exp(-0.5*(x - mu)**2 / (sigma**2)) * 1 / (math.sqrt(2 * np.pi * (sigma**2)))

        return gaussian

    def _measurement_probability(self, measurements: List[float], particle: Tuple[float, float, float], sensor_list: List[int] = range(16)) -> float:
        """Computes the probability of a set of measurements given a particle's pose.

        If a measurement is unavailable (usually because it is out of range), it is replaced with twice the sensor range
        to perform the computation. This value has experimentally been proven valid to deal with missing measurements.
        Nevertheless, it might not be the optimal replacement value.

        Args:
            measurements: Sensor measurements [m].
            particle: Particle pose (x, y, theta) in [m] and [rad].
            sensor_list: Sensor list to compute the measurement probability.

        Returns:
            float: Probability.

        """
        pred_measurements = self._sense(particle, sensor_list)
        p = 1
        measurement_list = []

        for i, measure in enumerate(measurements):
            if i in sensor_list:
                measurement_list.append(measure)

        for i, measure in enumerate(measurement_list):
            if measure == float('inf'):
                measure = 1.2 * self._sensor_range
            if pred_measurements[i] == float('inf'):
                pred_measurements[i] = 1.2 * self._sensor_range
            p = p * self._gaussian(measure, self._sense_noise, pred_measurements[i])

        return p

    def _sensor_rays(self, particle: Tuple[float, float, float]) -> List[List[Tuple[float, float]]]:
        """Determines the simulated sensor ray segments for a given particle.

        Args:
            particle: Particle pose (x, y, theta) in [m] and [rad].

        Returns: Ray segments.
                 Format: [[(x0_begin, y0_begin), (x0_end, y0_end)], [(x1_begin, y1_begin), (x1_end, y1_end)], ...]

        """
        x = particle[0]
        y = particle[1]
        theta = particle[2]

        # Convert sensors to world coordinates
        xw = [x + ds * math.cos(theta + phi) for ds, phi in zip(self._ds, self._phi)]
        yw = [y + ds * math.sin(theta + phi) for ds, phi in zip(self._ds, self._phi)]
        tw = [sensor[2] for sensor in self._sensors]

        rays = []

        for xs, ys, ts in zip(xw, yw, tw):
            x_end = xs + self._sensor_range * math.cos(theta + ts)
            y_end = ys + self._sensor_range * math.sin(theta + ts)
            rays.append([(xs, ys), (x_end, y_end)])

        return rays


def test():
    """Function used to test the ParticleFilter class independently."""
    import time
    from robot_p3dx import RobotP3DX

    # Measurements from sensors 1 to 8 [m]
    measurements = [
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.9343, float('inf'), float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8430, float('inf'), float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8430, float('inf'), float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8430, 0.8582, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8430, 0.7066, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8430, 0.5549, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8430, 0.4957, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8430, 0.4957, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8430, 0.4957, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8430, 0.4957, float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8430, 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8430, 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), float('inf'), 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), float('inf'), 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), float('inf'), 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), float('inf'), 0.4957, 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), float('inf'), float('inf'), 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), float('inf'), float('inf'), 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), float('inf'), float('inf'), 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), float('inf'), float('inf'), 0.3619),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.9920, float('inf'), float('inf')),
        (0.3618, 0.4895, 0.8337, float('inf'), float('inf'), 0.8795, float('inf'), float('inf')),
        (0.3832, 0.6021, float('inf'), float('inf'), 1.2914, 0.9590, float('inf'), float('inf')),
        (0.4207, 0.7867, float('inf'), float('inf'), 0.9038, float('inf'), float('inf'), 0.5420),
        (0.4778, float('inf'), float('inf'), float('inf'), 0.8626, float('inf'), float('inf'), 0.3648),
        (0.5609, float('inf'), float('inf'), 0.9514, 0.9707, float('inf'), float('inf'), 0.3669),
        (0.6263, float('inf'), float('inf'), 0.8171, 0.8584, float('inf'), float('inf'), 0.4199),
        (0.6918, float('inf'), 0.9942, 0.6828, 0.7461, float('inf'), float('inf'), 0.5652),
        (0.7572, 0.9544, 0.9130, 0.5485, 0.6338, float('inf'), float('inf'), 0.7106),
        (0.8226, 0.8701, 0.8319, 0.4142, 0.5215, float('inf'), float('inf'), 0.8559),
        (0.8880, 0.7858, 0.7507, 0.2894, 0.4092, float('inf'), float('inf'), float('inf')),
        (0.9534, 0.7016, 0.6696, 0.2009, 0.2969, float('inf'), float('inf'), float('inf')),
        (float('inf'), 0.6173, 0.5884, 0.1124, 0.1847, 0.4020, float('inf'), float('inf')),
        (0.9789, 0.5330, 0.1040, 0.0238, 0.0724, 0.2183, float('inf'), float('inf'))]

    # Wheel angular speed commands (left, right) [rad/s]
    motions = [(0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1),
               (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (0.5, 0), (0.5, 0), (0.5, 0), (0.5, 0),
               (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    dt = 1  # Sampling time [s]

    m = Map('map_pf.json', sensor_range=RobotP3DX.SENSOR_RANGE, compiled_intersect=True, use_regions=True)
    pf = ParticleFilter(m, RobotP3DX.SENSORS[:8], RobotP3DX.SENSOR_RANGE, particle_count=500)

    for u, z in zip(motions, measurements):
        # Solve differential kinematics
        v = (u[0] + u[1]) * RobotP3DX.WHEEL_RADIUS / 2
        w = (u[1] - u[0]) * RobotP3DX.WHEEL_RADIUS / RobotP3DX.TRACK

        # Move
        start = time.time()
        pf.move(v, w, dt)
        move = time.time() - start

        start = time.time()
        pf.show('Move', save_figure=True)
        plot_move = time.time() - start

        # Sense
        start = time.time()
        pf.resample(z)
        pf.get_current_position()
        sense = time.time() - start

        start = time.time()
        pf.show('Sense', save_figure=True)
        plot_sense = time.time() - start

        # Display timing results
        print('Particle filter: {0:6.3f} s  =  Move: {1:6.3f} s  +  Sense: {2:6.3f} s   |   Plotting: {3:6.3f} s  =  Move: {4:6.3f} s  +  Sense: {5:6.3f} s'.format(move + sense, move, sense, plot_move + plot_sense, plot_move, plot_sense))


# This "strange" function is only called if this script (particle_filter.py) is the program's entry point.
if __name__ == '__main__':
    try:
        test()
    except KeyboardInterrupt:  # Press Ctrl+C to gracefully stop the program
        pass
