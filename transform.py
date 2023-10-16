import numpy as np
from vector import Vector3

def x_rotation_matrix(rot):
    """Creates a 3x3 rotation of rot degrees (in radians) CCW around the X axis"""
    mat = np.identity(3)
    mat[1, 1] = np.cos(rot)
    mat[1, 2] = -np.sin(rot)
    mat[2, 1] = np.sin(rot)
    mat[2, 2] = np.cos(rot)
    return mat


def y_rotation_matrix(rot):
    """Creates a 3x3 rotation of rot degrees (in radians) CCW around the Y axis"""
    mat = np.identity(3)
    mat[0, 0] = np.cos(rot)
    mat[0, 2] = np.sin(rot)
    mat[2, 0] = -np.sin(rot)
    mat[2, 2] = np.cos(rot)
    return mat


def z_rotation_matrix(rot):
    """Creates a 3x3 rotation of rot degrees (in radians) CCW around the Z axis"""
    mat = np.identity(3)
    mat[0, 0] = np.cos(rot)
    mat[0, 1] = -np.sin(rot)
    mat[1, 0] = np.sin(rot)
    mat[1, 1] = np.cos(rot)
    return mat


class Transform:
    def __init__(self):
        # Store the transform as the transformation matrix
        # Default to the identity matrix (no translation or rotation)
        self.mat = np.identity(4)

    def __repr__(self):
        return str(self.mat)

    def set_position(self, x, y, z):
        self.mat[0, 3] = x
        self.mat[1, 3] = y
        self.mat[2, 3] = z

    def set_rotation(self, x, y, z):
        # Apply the rotations in XYZ order
        mat1 = np.matmul(x_rotation_matrix(np.radians(x)), y_rotation_matrix(np.radians(y)))
        rotation = np.matmul(mat1, z_rotation_matrix(np.radians(z)))
        self.mat[0:3, 0:3] = rotation

    def transformation_matrix(self):
        return self.mat
    def inverse_matrix(self):
        rotation = np.transpose(self.mat[0:3, 0:3])
        position = np.matmul(rotation * -1, self.mat[0:3, 3])
        mat = np.identity(4)
        mat[0:3, 0:3] = rotation
        mat[0:3, 3] = position

        return mat

    def apply_to_point(self, p):
        p = np.append(p, 1)
        p = p.transpose()

        return np.matmul(self.mat, p)[:3]

    def apply_inverse_to_point(self, p):
        p = np.append(p, 1)
        p = p.transpose()

        return Vector3.from_array(np.matmul(self.inverse_matrix(), p)[:3])

    def apply_to_normal(self, n):
        n = np.append(n, 0)
        n = n.transpose()

        return np.matmul(self.mat, n)[:3]
