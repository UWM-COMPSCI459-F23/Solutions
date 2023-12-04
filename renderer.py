import numpy as np

from vector import Vector3


def barycentric_coordinates_2d(v0, v1, v2, p):
    numerator = ((v0[2] - v1[2]) * p[0]) + ((v1[0] - v0[0]) * p[2]) + (v0[0] * v1[2]) - (v1[0] * v0[2])
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
    def __init__(self, screen, camera, meshes, light):
        self.screen = screen
        self.camera = camera
        self.meshes = meshes
        self.light = light

    def render(self, shading="flat", \
               bg_color=[100, 100, 100], ambient_light=[1.0, 1.0, 1.0]):

        # Verify that screen buffer and camera ratio match
        if abs(self.screen.ratio() - self.camera.ratio()) > 0.001:
            print(self.screen.ratio(), self.camera.ratio())
            raise Exception("Screen buffer and camera are not the same ratio")

        image_buffer = np.full((self.screen.height, self.screen.width, 3), bg_color)
        z_buffer = np.full((self.screen.height, self.screen.width), 1000.0)

        min_depth = 1000.0
        max_depth = -1000.0

        for mesh in self.meshes:
            verts = [self.camera.project_point(mesh.transform.apply_to_point(p)) for p in mesh.verts]
            for vert in verts:
                if vert[1] < min_depth:
                    min_depth = vert[1]
                elif vert[1] > max_depth:
                    max_depth = vert[1]

        for mesh in self.meshes:
            # Get list of verts transformed to screen space
            verts_screen = [self.camera.project_point(mesh.transform.apply_to_point(p)) for p in mesh.verts]
            verts_world = [Vector3.from_array(mesh.transform.apply_to_point(p)) for p in mesh.verts]

            for i, face in enumerate(mesh.faces):
                # Normal culling
                normal = mesh.transform.apply_to_normal(mesh.normals[i])
                camera_forward = self.camera.transform.apply_to_normal(np.array([0, 1, 0]))
                if Vector3.dot(normal, camera_forward) >= 0.0:
                    continue

                # Get the screen space verts for the current face
                v0 = verts_screen[face[0]]
                v1 = verts_screen[face[1]]
                v2 = verts_screen[face[2]]

                # Find the rectangular bounds of triangle in screen space
                min_x = min(min(v0[0], v1[0]), v2[0])
                min_z = min(min(v0[2], v1[2]), v2[2])
                max_x = max(max(v0[0], v1[0]), v2[0])
                max_z = max(max(v0[2], v1[2]), v2[2])

                # Convert to pixel space
                top_left = self.screen.screen_to_pixel(min_x, max_z)
                bottom_right = self.screen.screen_to_pixel(max_x, min_z)

                # Flat shading only needs to do the color calculation once per face (not per fragment)
                if shading == "flat":
                    # Calculate point in world space
                    p_world = (verts_world[face[0]] * (1 / 3.0)) + (verts_world[face[1]] * (1 / 3.0)) + (
                                verts_world[face[2]] * (1 / 3.0))

                    # Get light vector
                    l = p_world.direction_to(self.light.transform.get_position())

                    # Cosine of angle between normal and light vector
                    cos_theta = max(0, Vector3.dot(normal, l))

                    d = p_world.distance_to(self.light.transform.get_position())

                    irradiance = self.light.color * (self.light.intensity * cos_theta / (d ** 2))

                    lambertian = np.minimum(1.0, (mesh.kd * mesh.diffuse_color / np.pi))

                    r = lambertian[0] * irradiance[0]
                    g = lambertian[1] * irradiance[1]
                    b = lambertian[2] * irradiance[2]

                    r = min(1.0, r + ambient_light[0] * mesh.ka)
                    g = min(1.0, g + ambient_light[1] * mesh.ka)
                    b = min(1.0, b + ambient_light[2] * mesh.ka)

                    flat_color = [int(r * 255), int(g * 255), int(b * 255)]

                # Iterate over the pixels (fragment shader)
                for x in range(max(0, top_left[0]), min(self.screen.width, bottom_right[0] + 1)):
                    for y in range(max(0, bottom_right[1]), min(self.screen.height, top_left[1] + 1)):

                        # Convert the pixel back to screen space
                        p_screen = self.screen.pixel_to_screen(x, y)

                        # Get the barycentric coordinates
                        alpha, beta, gamma = barycentric_coordinates_2d(v0, v1, v2, p_screen)

                        # Calculate y by barycentric interpolation
                        depth = (alpha * v0[1]) + (beta * v1[1]) + (gamma * v2[1])
                        p_screen[1] = depth

                        # Only continue if point is in CVV
                        if depth > 1.0 or depth < -1.0:
                            continue

                        # Only continue if p_screen is closer than value in z-buffer
                        if depth >= z_buffer[x, y]:
                            continue

                        # If p lies on the face
                        if (0.0 <= alpha <= 1.0) and (0.0 <= beta <= 1.0) and (0.0 <= gamma <= 1.0):
                            # Create a default color
                            color = [0, 0, 0]

                            if shading == "depth":
                                hue = int((depth - min_depth) / (max_depth - min_depth) * 127)
                                color = [hue, hue, hue]

                            if shading == "barycentric":
                                red = int(alpha * 255)
                                blue = int(beta * 255)
                                green = int(gamma * 255)
                                color = [red, blue, green]

                            if shading == "normal":
                                if np.array_equal(mesh.normals[i], [1.0, 0, 0]):
                                    color = [255, 0, 0]
                                elif np.array_equal(mesh.normals[i], [-1.0, 0, 0]):
                                    color = [127, 0, 0]
                                elif np.array_equal(mesh.normals[i], [0, 1.0, 0]):
                                    color = [0, 255, 0]
                                elif np.array_equal(mesh.normals[i], [0, -1.0, 0]):
                                    color = [0, 127, 0]
                                elif np.array_equal(mesh.normals[i], [0, 0, 1.0]):
                                    color = [0, 0, 255]
                                elif np.array_equal(mesh.normals[i], [0, 0, -1.0]):
                                    color = [0, 0, 127]

                            if shading == "flat":
                                color = flat_color

                            image_buffer[x, y] = color
                            z_buffer[x, y] = depth

            self.screen.draw(image_buffer)
        print("Done")