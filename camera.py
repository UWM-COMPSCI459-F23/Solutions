import numpy as np

from transform import Transform
from vector import Vector3


class OrthoCamera:
    def __init__(self, left, right, bottom, top, near, far):
        self.transform = Transform()
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far

        # Orthographic view transform for a right-handed Z-up coordinate system
        self.ortho_transform = np.identity(4)
        self.ortho_transform[0] = np.array([2 / (right - left), 0, 0, -((right + left) / (right - left))])
        self.ortho_transform[1] = np.array([0, 2 / (far - near), 0, -((far + near) / (far - near))])
        self.ortho_transform[2] = np.array([0, 0, 2 / (top - bottom), -((top + bottom) / (top - bottom))])

        # np.linalg.inv(self.ortho_transform) also works
        rotation = np.identity(3)
        rotation[0, 0] = 1.0 / self.ortho_transform[0, 0]
        rotation[1, 1] = 1.0 / self.ortho_transform[1, 1]
        rotation[2, 2] = 1.0 / self.ortho_transform[2, 2]
        position = np.matmul(rotation * -1, self.ortho_transform[0:3, 3])
        self.inverse_ortho_transform = np.identity(4)
        self.inverse_ortho_transform[0:3, 0:3] = rotation
        self.inverse_ortho_transform[0:3, 3] = position

    def ratio(self):
        return abs(self.right - self.left) / abs(self.top - self.bottom)

    def project_point(self, p_world):
        p_camera = self.transform.apply_inverse_to_point(p_world)
        p_screen = np.matmul(self.ortho_transform, np.append(p_camera, 1).transpose())

        return Vector3.from_array(p_screen[:3])

    def inverse_project_point(self, p_screen):
        p_camera = np.matmul(self.inverse_ortho_transform, np.append(p_screen, 1).transpose())[:3]
        p_world = self.transform.apply_to_point(p_camera)

        return Vector3.from_array(p_world)