"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""

import numpy as np
import math
from heapq import heappush, heappop


class AStarPlanner:
    """
    A* path planner for 3D space with cylindrical obstacles.
    
    The planner discretizes the 3D space into a grid and uses A* algorithm
    to find collision-free paths from start to goal positions.
    """
    
    def __init__(self, env, grid_resolution=0.5):
        """
        Initialize the A* planner.
        
        Args:
            env: FlightEnvironment object with is_collide() and is_outside() methods
            grid_resolution: Resolution of the discretized grid (default: 0.5m)
        """
        self.env = env
        self.grid_resolution = grid_resolution
        self.env_bounds = [(0, 20), (0, 20), (0, 5)]  # (x, y, z) bounds
        
    def _discretize_position(self, pos):
        """Convert continuous position to grid index."""
        x, y, z = pos
        ix = int(x / self.grid_resolution)
        iy = int(y / self.grid_resolution)
        iz = int(z / self.grid_resolution)
        return (ix, iy, iz)
    
    def _undiscretize_position(self, idx):
        """Convert grid index back to continuous position."""
        ix, iy, iz = idx
        x = (ix + 0.5) * self.grid_resolution
        y = (iy + 0.5) * self.grid_resolution
        z = (iz + 0.5) * self.grid_resolution
        return (x, y, z)
    
    def _is_valid_node(self, idx):
        """Check if a grid node is valid (within bounds and collision-free)."""
        ix, iy, iz = idx
        
        # Check bounds
        x_max = int(self.env_bounds[0][1] / self.grid_resolution)
        y_max = int(self.env_bounds[1][1] / self.grid_resolution)
        z_max = int(self.env_bounds[2][1] / self.grid_resolution)
        
        if not (0 <= ix < x_max and 0 <= iy < y_max and 0 <= iz < z_max):
            return False
        
        # Check collision
        pos = self._undiscretize_position(idx)
        if self.env.is_collide(pos, epsilon=0.2) or self.env.is_outside(pos):
            return False
        
        return True
    
    def _get_neighbors(self, idx):
        """Get valid neighboring nodes (6-connected: up, down, left, right, forward, back)."""
        ix, iy, iz = idx
        neighbors = []
        
        # 6-connectivity (also considers diagonal movements in xy plane for efficiency)
        directions = [
            (1, 0, 0), (-1, 0, 0),  # x-axis
            (0, 1, 0), (0, -1, 0),  # y-axis
            (0, 0, 1), (0, 0, -1),  # z-axis
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),  # xy diagonals
        ]
        
        for dx, dy, dz in directions:
            neighbor = (ix + dx, iy + dy, iz + dz)
            if self._is_valid_node(neighbor):
                neighbors.append(neighbor)
        
        return neighbors
    
    def _heuristic(self, current, goal):
        """Euclidean distance heuristic."""
        dx = goal[0] - current[0]
        dy = goal[1] - current[1]
        dz = goal[2] - current[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _get_distance(self, idx1, idx2):
        """Calculate distance between two grid nodes."""
        dx = idx2[0] - idx1[0]
        dy = idx2[1] - idx1[1]
        dz = idx2[2] - idx1[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def plan(self, start, goal):
        """
        Find a collision-free path from start to goal using A* algorithm.
        
        Args:
            start: Tuple (x, y, z) for start position
            goal: Tuple (x, y, z) for goal position
            
        Returns:
            numpy array of shape (N, 3) containing the path, or None if no path found
        """
        # Check if start and goal are valid
        if self.env.is_collide(start, epsilon=0.2) or self.env.is_outside(start):
            print("ERROR: Start position is in collision or outside bounds")
            return None
        
        if self.env.is_collide(goal, epsilon=0.2) or self.env.is_outside(goal):
            print("ERROR: Goal position is in collision or outside bounds")
            return None
        
        start_idx = self._discretize_position(start)
        goal_idx = self._discretize_position(goal)
        
        # A* implementation
        open_set = []
        closed_set = set()
        g_score = {start_idx: 0}
        f_score = {start_idx: self._heuristic(start_idx, goal_idx)}
        parent = {start_idx: None}
        
        heappush(open_set, (f_score[start_idx], start_idx))
        
        while open_set:
            _, current = heappop(open_set)
            
            if current == goal_idx:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(self._undiscretize_position(node))
                    node = parent[node]
                path.reverse()
                return np.array(path)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + self._get_distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal_idx)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        print("ERROR: No path found from start to goal")
        return None
            











