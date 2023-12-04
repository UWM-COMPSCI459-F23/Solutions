import stl

from vector import Vector3
from transform import Transform

class Mesh:
    def __init__(self, diffuse_color=np.array([1.0, 0.0, 1.0]), \
                 specular_color=np.array([1.0, 1.0, 1.0]), ka=0.2, kd=1.0, ks=0.2, ke=100):
        # Positions of vertex in local space (x, y, z)
        self.verts = []

        # List of 3 vertex indices that make face
        self.faces = []

        # Face normal, calculated using right-hand curl
        self.normals = []

        # Rectangular bounds
        self.bounds = ()

        # Identity transform
        self.transform = Transform()

        # The mesh's material properties (maybe move to a separate class?)
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.ke = ke

    @staticmethod
    def from_stl(stl_path,diffuse_color,specular_color,ka,kd,ks,ke):
        """Reads the stl file at stl_path and converts it to our own vertex list format"""

        # Use numpy-stl to read the STL file
        mesh = stl.mesh.Mesh.from_file(stl_path)

        verts = []
        faces = []
        normals = []

        # Initiate bounds
        min_x = mesh.vectors[0, 0, 0]
        min_y = mesh.vectors[0, 0, 1]
        min_z = mesh.vectors[0, 0, 2]
        max_x = mesh.vectors[0, 0, 0]
        max_y = mesh.vectors[0, 0, 1]
        max_z = mesh.vectors[0, 0, 2]

        for face in mesh.vectors:
            cur_face = []
            for vertex in face:
                # This checks to see if the vertex is already in our list, eliminating duplicates
                l = [v[0] == vertex[0] and v[1] == vertex[1] and v[2] == vertex[2] for v in verts]
                if any(l):
                    for i, val in enumerate(l):
                        if val:
                            index = i
                            break

                # If the vertex is not in the list, add  it
                else:
                    index = len(verts)
                    verts.append(Vector3.from_array(vertex))

                    # Update bounds
                    min_x = min(vertex[0], min_x)
                    min_y = min(vertex[1], min_y)
                    min_z = min(vertex[2], min_z)
                    max_x = min(vertex[0], max_x)
                    max_y = min(vertex[1], max_y)
                    max_z = min(vertex[2], max_z)

                cur_face.append(index)

            # Calculate the face normal
            # You could alternatively just grab the already computed normals from the STL file
            a = Vector3.from_array(verts[cur_face[0]])
            b = Vector3.from_array(verts[cur_face[1]])
            c = Vector3.from_array(verts[cur_face[2]])
            n = Vector3.cross(b - a, c - a)

            # Add face
            faces.append(cur_face)
            normals.append(n.normalized())


        # Instantiate the mesh and add the componenents
        mesh = Mesh(diffuse_color, specular_color, ka, kd, ks, ke)
        mesh.verts = verts
        mesh.faces = faces
        mesh.normals = normals
        mesh.bounds = ((min_x, max_x), (min_y, max_y), (min_z, max_z))

        return mesh

