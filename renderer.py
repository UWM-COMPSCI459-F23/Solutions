import numpy as np

from vector import Vector3


def barycentric_coordinates_2d(v0, v1, v2, p):
    numerator =   ((v0[2] - v1[2]) * p[0] ) + ((v1[0] - v0[0]) * p[2] ) + (v0[0] * v1[2]) - (v1[0] * v0[2])
    denominator = ((v0[2] - v1[2]) * v2[0]) + ((v1[0] - v0[0]) * v2[2]) + (v0[0] * v1[2]) - (v1[0] * v0[2])
    if denominator == 0.0:
        return -1, -1, -1
    gamma = numerator / denominator

    numerator = ((v0[2] - v2[2]) * p[0]) + ((v2[0] - v0[0]) * p[2]) + (v0[0] * v2[2]) - (v2[0] * v0[2])
    denominator = ((v0[2] - v2[2]) * v1[0]) + ((v2[0] - v0[0]) * v1[2]) + (v0[0] * v2[2]) - (v2[0] * v0[2])

    if denominator == 0.0:
        return -1, -1, -1
    beta = numerator / denominator

    alpha = 1 - beta - gamma

    return alpha, beta, gamma

class Renderer:
    def __init__(self, screen, camera, mesh):
        self.screen = screen
        self.camera = camera
        self.mesh = mesh

    def render(self, bg_color):

        # Verify that screen buffer and camera ratio match
        if abs(self.screen.ratio() - self.camera.ratio()) > 0.001:
            print(self.screen.ratio(), self.camera.ratio())
            raise Exception("Screen buffer and camera are not the same ratio")

        image_buffer = np.full((self.screen.height, self.screen.width, 3), bg_color)

        # Get list of verts transformed to camera space
        verts = [self.camera.project_point(self.mesh.transform.apply_to_point(p)) for p in self.mesh.verts]

        for i, face in enumerate(self.mesh.faces):
            # Normal culling
            normal = self.mesh.transform.apply_to_normal(self.mesh.normals[i])
            camera_forward = self.camera.transform.apply_to_normal(np.array([0, 1, 0]))
            if Vector3.dot(normal, camera_forward) < 0:
                # if np.dot(normal, camera_forward) < 0:
                continue

            v0 = verts[face[0]]
            v1 = verts[face[1]]
            v2 = verts[face[2]]

            # Find the square bounds of triangle in screen space
            min_x = min(min(v0[0], v1[0]), v2[0])
            min_y = min(min(v0[2], v1[2]), v2[2])
            max_x = max(max(v0[0], v1[0]), v2[0])
            max_y = max(max(v0[2], v1[2]), v2[2])

            # Convert to pixel space
            top_left = self.screen.screen_to_pixel(min_x, max_y)
            bottom_right = self.screen.screen_to_pixel(max_x, min_y)

            # Iterate over the pixels (fragment shader)
            for x in range(max(0, top_left[0]), min(self.screen.width, bottom_right[0] + 1)):
                for y in range(max(0, bottom_right[1]), min(self.screen.height, top_left[1] + 1)):

                    # Convert the pixel back to screen space
                    p = self.screen.pixel_to_screen(x, y)

                    # Get the barymetric coordinates
                    alpha, beta, gamma = barycentric_coordinates_2d(v0, v1, v2, p)

                    # If p lies on the face
                    if (0.0 <= alpha <= 1.0) and (0.0 <= beta <= 1.0) and (0.0 <= gamma <= 1.0):
                        image_buffer[x, y] = [0, 0, 0]

        self.screen.draw(image_buffer)
        print("Done")