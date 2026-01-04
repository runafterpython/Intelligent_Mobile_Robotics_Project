"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


class PolynomialTrajectoryGenerator:
	"""
	Generate time-parameterized polynomial trajectories for x(t), y(t), z(t)
	that interpolate the given waypoints. Uses cubic spline interpolation
	with time allocation proportional to segment length.
	"""

	def __init__(self, nominal_speed=1.0, min_dt=0.5):
		"""
		Args:
			nominal_speed: nominal speed (m/s) used to allocate time between waypoints
			min_dt: minimum time for any segment (seconds)
		"""
		self.nominal_speed = nominal_speed
		self.min_dt = min_dt

	def _allocate_times(self, path):
		# path: (N,3)
		diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
		# avoid zero distances
		diffs[diffs == 0] = 1e-6
		times = diffs / self.nominal_speed
		times = np.maximum(times, self.min_dt)
		t = np.concatenate(([0.0], np.cumsum(times)))
		return t

	def generate(self, path, sample_dt=0.02):
		"""
		Generate trajectory samples for the given path.

		Args:
			path: iterable of shape (N,3)
			sample_dt: time step for sampled output

		Returns:
			t_samples, x_samples, y_samples, z_samples, waypoint_times
		"""
		path = np.asarray(path, dtype=float)
		if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
			raise ValueError("path must be an (N,3) array with N>=2")

		waypoint_times = self._allocate_times(path)

		# Fit cubic splines for each axis
		cs_x = CubicSpline(waypoint_times, path[:, 0], bc_type='natural')
		cs_y = CubicSpline(waypoint_times, path[:, 1], bc_type='natural')
		cs_z = CubicSpline(waypoint_times, path[:, 2], bc_type='natural')

		t_samples = np.arange(waypoint_times[0], waypoint_times[-1] + 1e-8, sample_dt)
		x_samples = cs_x(t_samples)
		y_samples = cs_y(t_samples)
		z_samples = cs_z(t_samples)

		return t_samples, x_samples, y_samples, z_samples, waypoint_times

	def generate_and_plot(self, path, sample_dt=0.02):
		t, x, y, z, waypoint_times = self.generate(path, sample_dt=sample_dt)

		fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
		axes[0].plot(t, x, '-b')
		axes[0].scatter(waypoint_times, np.asarray(path)[:, 0], c='r')
		axes[0].set_ylabel('x (m)')
		axes[0].grid(True)

		axes[1].plot(t, y, '-b')
		axes[1].scatter(waypoint_times, np.asarray(path)[:, 1], c='r')
		axes[1].set_ylabel('y (m)')
		axes[1].grid(True)

		axes[2].plot(t, z, '-b')
		axes[2].scatter(waypoint_times, np.asarray(path)[:, 2], c='r')
		axes[2].set_ylabel('z (m)')
		axes[2].set_xlabel('time (s)')
		axes[2].grid(True)

		plt.tight_layout()
		plt.show()


