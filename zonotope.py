import itertools
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class zono:

    def __init__(self, values: np.array = None, dimension = 1, generators = 1) -> None:
        """
        Initialize a zonotope.
        If values is None, a zonotope with dimension and generators is created.
        If values is not None, it must be a numpy array with shape (dimension, generators + 1).
        The first column is the center of the zonotope.
        The remaining columns are the generators.
        The zonotope is assumed to be closed.
        
        Args:
        values: numpy array with shape (dimension, generators + 1).
        dimension: dimension of the zonotope.
        generators: number of generators.
        """
        if values is None:
            self.values = np.zeros((dimension, generators))
        elif type(values) is list:
            self.values = np.array(values)
        else:
            self.values = values
        self.dimensions = self.values.shape[0]
        self.generators = self.values.shape[1] - 1
    
    def __str__(self) -> str:
        return str(self.values)
    
    def __repr__(self) -> str:
        return str(self.values)
    
    def __add__(self, other: 'zono') -> 'zono':
        """
        Add two zonotopes with the same dimension. 
        If the dimension is different, an value error is raised.
        The zonootopes don't have to have the same number of generators. 
        If they don't, the generators are padded with zeros.
        
        Args:
            other: zonotope to add.
        
        Returns:
            zonotope sum.
        """
        if self.values.shape[0] != other.values.shape[0]:
            raise ValueError("Dimension mismatch")
        if self.values.shape[1] == other.values.shape[1]:
            return zono(np.add(self.values, other.values))
        if self.values.shape[1] > other.values.shape[1]:
            v = np.pad(other.values, [(0, 0), (0, self.values.shape[1] - other.values.shape[1])], 'constant', constant_values = 0)
            return zono(np.add(self.values, v))
        else:
            v = np.pad(self.values, [(0, 0), (0, other.values.shape[1] - self.values.shape[1])], 'constant', constant_values = 0)
            return zono(np.add(v, other.values))
    
    def __mul__(self, other: float) -> 'zono':
        return zono(np.multiply(self.values, other))
    
    def __mul__(self, other: int) -> 'zono':
        return zono(np.multiply(self.values, other))
    
    def __rmul__(self, other: float) -> 'zono':
        return self.__mul__(other)
    
    def __rmul__(self, other: int) -> 'zono':
        return self.__mul__(other)
    
    def combine(self, other: 'zono') -> 'zono':
        """
        Combine two zonotopes.

        Args:
            other: zonotope to combine.
        """
        if self.values.shape[1] == other.values.shape[1]:
            return zono(values = np.append(self.values, other.values, axis=0))
        if self.values.shape[1] > other.values.shape[1]:
            v = np.pad(other.values, [(0, 0), (0, self.values.shape[1] - other.values.shape[1])], 'constant', constant_values = 0)
            return zono(values = np.append(self.values, v, axis=0))
        else:
            v = np.pad(self.values, [(0, 0), (0, other.values.shape[1] - self.values.shape[1])], 'constant', constant_values = 0)
            return zono(values = np.append(v, other.values, axis=0))
    
    def split(self) -> 'zono':
        """
        Split a zonotope into two zonotopes.
        The number of dimensions must be even.
        The zonotopes are assumed to be closed.

        Returns:
            two zonotopes with the same size.
        """
        if self.dimensions % 2 != 0:
            raise ValueError("Dimension must be even")
        else:
            values = np.split(self.values, 2, axis=0)
            return zono(values[0]), zono(values[1])
    
    def upper_bound(self, dimension = 1) -> float:
        """
        Returns the upper bound of the zonotope in the given dimension.
        
        Args:
            dimension: dimension of the upper bound.
        
        Returns:
            upper bound in the given dimension.
        """
        bound = self.values[dimension-1][0]
        for g in self.values[dimension-1][1:]:
            if g > 0:
                bound += g
            else:
                bound -= g
        return bound
    
    def lower_bound(self, dimension = 1) -> float:
        """
        Returns the lower bound of the zonotope in the given dimension.
        
        Args:
            dimension: dimension of the lower bound.
        
        Returns:
            lower bound in the given dimension."""
        bound = self.values[dimension-1][0]
        for g in self.values[dimension-1][1:]:
            if g > 0:
                bound -= g
            else:
                bound += g
        return bound
    
    def to_intervals(self):
        """
        Returns the intervals of the zonotope.

        Returns:
            list of intervals tuples.
        """
        intervals = []
        for i in range(self.dimensions):
            intervals.append((self.lower_bound(i + 1), self.upper_bound(i + 1)))
        return intervals
    
    def get_random_point(self):
        """
        Returns a random point in the zonotope.
        
        Returns:
            random point.
        """
        point = []
        for i in range(self.dimensions):
            p = self.values[i][0]
            for g in self.values[i][1:]:
                p += g * np.random.uniform(-1, 1)
            point.append(p)
        return point
    
    def check_random_point(self, point: list):
        """
        Checks if the given point is in the zonotope.
        
        Args:
            point: point to check.
        
        Returns:
            True if the point is in the zonotope, False otherwise.
        """
        if len(point) != self.dimensions:
            raise ValueError("Dimension mismatch")
        for i in range(self.dimensions):
            if point[i] < self.lower_bound(i + 1) or point[i] > self.upper_bound(i + 1):
                return False
        return True
    
    def visualize(self, quiver = False, shape = False, shape_color = "g", fig = None, ax = None) -> None:
        """
        Visualize the zonotope.
        If quiver is True, the generators are visualized as quiver plots.
        If shape is True, the zonotope is visualized as a polygon.
        If fig and ax are not None, the zonotope is visualized in the given figure and axis.
        If fig and ax are None, a new figure and axis are created and the plot is drawn.

        Args:
            quiver: boolean.
            shape: boolean.
            fig: figure.
            ax: axis.
        
        Returns:
            None.
        """
        show_self = (fig == None) or (ax == None)
        if self.dimensions > 3:
            raise ValueError("Dimension must be <= 3")
        else:
            if fig is None:
                fig = plt.figure()
            if self.dimensions == 1:
                if ax is None:
                    ax = fig.add_subplot(111, aspect = 'equal')
                if quiver: #TODO implement 1D quiver visualization
                    pass
                if shape:
                    raise NotImplementedError("Shape visualization not implemented for 1D zonotopes")
            elif self.dimensions == 2:
                if ax is None:
                    ax = fig.add_subplot(111, aspect = 'equal')
                if quiver:
                    for i in range(self.generators):
                        ax.quiver(self.values[0][0], self.values[1][0], self.values[0][i + 1], self.values[1][i + 1])
                        ax.quiver(self.values[0][0], self.values[1][0], -self.values[0][i + 1], -self.values[1][i + 1])
                if shape and self.generators > 1:
                    x = []
                    for i in itertools.product([-1.0, 1.0], repeat = self.generators):
                        x.append((np.sum(self.values[0][1:] * np.array(list(i))) + self.values[0][0]))
                    y = []
                    for i in itertools.product([-1.0, 1.0], repeat = self.generators):
                        y.append((np.sum(self.values[1][1:] * np.array(list(i))) + self.values[1][0]))
                    x, y = np.array(x), np.array(y)
                    order = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
                    ax.fill(x[order], y[order], shape_color, alpha=0.5)
            else:
                if ax is None:
                    ax = Axes3D(fig)
                if quiver: #TODO implement 3D quiver visualization
                    pass
                if shape: #TODO Fix 3D shape visualization
                    x = []
                    for i in itertools.product([-1.0, 1.0], repeat = self.generators):
                        x.append((np.sum(self.values[0][1:] * np.array(list(i))) + self.values[0][0]))
                    y = []
                    for i in itertools.product([-1.0, 1.0], repeat = self.generators):
                        y.append((np.sum(self.values[1][1:] * np.array(list(i))) + self.values[1][0]))
                    z = []
                    for i in itertools.product([-1.0, 1.0], repeat = self.generators):
                        z.append((np.sum(self.values[2][1:] * np.array(list(i))) + self.values[2][0]))
                    print(f"X: {x}")
                    print(f"Y: {y}")
                    print(f"Z: {z}")
                    vertices = [list(zip(x,y,z))]
                    print(vertices)
                    ax.add_collection3d(Poly3DCollection(vertices))
            if show_self:
                plt.show()

    def from_file(file_name:str) -> 'zono':
        """
        Loads a zonotope from a file.
        
        Args:
            file_name: name of the file.
        
        Returns:
            zonotope loaded from the file.
        """
        with open(file_name, 'r') as f:
            lines = f.readlines()
        values = []
        for line in lines[2:]:
            values.append(list(map(float, line.split())))
        return zono(values=np.array(values))

if __name__ == "__main__":
    z = zono.from_file("src/test.txt")
    print(z)
