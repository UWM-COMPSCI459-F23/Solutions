import numpy as np


class Vector3:

    @staticmethod
    def zeros():
        return Vector3(0, 0, 0)

    @staticmethod
    def from_array(array):
        return Vector3(array[0], array[1], array[2])

    @staticmethod
    def dot(vec1, vec2):
        return np.dot(vec1.vec, vec2.vec)

    @staticmethod
    def cross(vec1, vec2):
        return Vector3.from_array(np.cross(vec1.vec, vec2.vec))

    def __init__(self, x, y, z):
        self.vec = np.array([x, y, z])

    def __getattr__(self, name):
        # Convenient way to get elements
        if name == 'x':
            return self.vec[0]
        if name == 'y':
            return self.vec[1]
        if name == 'z':
            return self.vec[2]

        # Otherwise, treat this object as if it were the vec
        else:
            return self.vec.__getattribute__(name)

    def __getitem__(self, index):
        return self.vec[index]

    def __add__(self, rhs):
        return Vector3.from_array(self.vec + rhs.vec)

    def __sub__(self, rhs):
        return Vector3.from_array(self.vec - rhs.vec)

    def __neg__(self):
        return Vector3.from_array(np.negative(self.vec))

    def __mul__(self, scalar):
        return Vector3.from_array(self.vec * scalar)

    def __div__(self, scalar):
        return Vector3.from_array(self.vec * scalar)

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def magnitude(self):
        return np.sqrt(np.sum(np.power(self.vec, 2)))

    def normalized(self):
        return Vector3.from_array(self.vec / self.magnitude())

    def distance_to(self, other):
        return (other - self).magnitude()

    def direction_to(self, other):
        return (other - self).normalized()

    def swap(self, index1, index2):
        self.vec[index1], self.vec[index2] = self.vec[index2], self.vec[index1]



