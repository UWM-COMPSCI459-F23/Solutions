import numpy as np
from transform import Transform

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
        self.ortho_transform[0] = np.array([2 / (right - left), 0,                0,                  -((right + left) / (right - left))])
        self.ortho_transform[1] = np.array([0,                  2 / (far - near), 0,                  -((near + far) / (far - near))    ])
        self.ortho_transform[2] = np.array([0,                  0,                2 / (top - bottom), -((top + bottom) / (top - bottom))])

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

        return p_screen[:3]

    def project_inverse_point(self, p_screen):
        #p_screen = np.array([p_screen[0], p_screen[2], p_screen[1]])
        p_camera = np.matmul(self.inverse_ortho_transform, np.append(p_screen, 1).transpose())[:3]
        p_world = self.transform.apply_to_point(p_camera)

        return p_world[:3]


class PerspectiveCamera:

    @staticmethod
    def from_FOV(fov, near, far, ratio):
        left = np.tan(np.radians(fov) / 2) * abs(near)
        right = -left
        top = ratio * left
        bottom = -top

        return PerspectiveCamera(left, right, bottom, top, near, far)

    def __init__(self, left, right, bottom, top, near, far):
        self.transform = Transform()
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far

        # need ortho transform matrix to generate perspective projection
        self.ortho_transform = np.identity(4)
        self.ortho_transform[0] = np.array([2 / (right - left), 0, 0, -((right + left) / (right - left))])
        self.ortho_transform[1] = np.array([0, 2 / (far - near), 0, -((near + far) / (far - near))])
        self.ortho_transform[2] = np.array([0, 0, 2 / (top - bottom), -((top + bottom) / (top - bottom))])

        rotation = np.identity(3)
        rotation[0, 0] = 1.0 / self.ortho_transform[0, 0]
        rotation[1, 1] = 1.0 / self.ortho_transform[1, 1]
        rotation[2, 2] = 1.0 / self.ortho_transform[2, 2]
        position = np.matmul(rotation * -1, self.ortho_transform[0:3, 3])
        self.inverse_ortho_transform = np.identity(4)
        self.inverse_ortho_transform[0:3, 0:3] = rotation
        self.inverse_ortho_transform[0:3, 3] = position

        # Orthographic view transform for a right-handed Z-up coordinate system
        self.persp_transform = np.identity(4)

        self.persp_transform[0] = np.array([near, 0, 0, 0])  # n 0 0 0
        self.persp_transform[1] = np.array([0, near + far, 0, -(far * near)])  # 0 n 0 0
        self.persp_transform[2] = np.array([0, 0, near, 0])  # 0 0 n+f -fn
        self.persp_transform[3] = np.array([0, 1, 0, 0])  # 0 0 1 0

        self.inverse_persp_transform = np.identity(4)

        self.inverse_persp_transform[0] = np.array([1 / near, 0, 0, 0])
        self.inverse_persp_transform[1] = np.array([0, 0, 0, 1])
        self.inverse_persp_transform[2] = np.array([0, 0, 1 / near, 0])
        self.inverse_persp_transform[3] = np.array([0, 1 / (far * near), 0, (far + near) / (far * near)])

        # self.inverse_persp_transform = np.matmul(self.inverse_persp_transform, inverse_ortho_transform)

    def ratio(self):
        return abs(self.right - self.left) / abs(self.top - self.bottom)

    def project_point(self, p_world):
        p_camera = self.transform.apply_inverse_to_point(p_world)

        p_screen = np.matmul(self.persp_transform, np.append(p_camera, 1).transpose())

        p_screen = p_screen / p_screen[3]

        p_screen = np.matmul(self.ortho_transform, p_screen)

        return p_screen[:3]

    def project_inverse_point(self, p_screen):
        p_screen = np.append(p_screen, 1.0).transpose()

        p_screen = np.matmul(self.inverse_ortho_transform, p_screen)

        y = (self.far * self.near) / (self.near + self.far - p_screen[1])

        p_screen = p_screen * y

        p_camera = np.matmul(self.inverse_persp_transform, p_screen)[:3]

        p_world = self.transform.apply_to_point(p_camera)

        return p_world[:3]
