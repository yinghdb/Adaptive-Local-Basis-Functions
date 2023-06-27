import os
from plyfile import PlyElement, PlyData
import numpy as np
import skimage.measure

def export_pointcloud(vertices, out_file, colors=None, as_text=True):
    assert(vertices.shape[1] == 3)
    vertices = vertices.astype(np.float32)
    vertices = np.ascontiguousarray(vertices)
    vector_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = vertices.view(dtype=vector_dtype).flatten()
    if colors is not None:
        assert(colors.shape[1] == 3)
        colors = colors.astype(np.uint8)
    plyel = PlyElement.describe(vertices, 'vertex')
    plydata = PlyData([plyel], text=as_text)
    plydata.write(out_file)

def save_ply(points, filename, colors=None, normals=None, as_text=True):
    vertex = np.core.records.fromarrays(points.transpose(), names='x, y, z', formats='f4, f4, f4')
    n = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.core.records.fromarrays(normals.transpose(), names='nx, ny, nz', formats='f4, f4, f4')
        assert len(vertex_normal) == n
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        vertex_color = np.core.records.fromarrays(colors.transpose() * 255, names='red, green, blue',
                                                  formats='u1, u1, u1')
        assert len(vertex_color) == n
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(n, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=as_text)
    if not os.path.exists(os.path.dirname(filename)):
       os.makedirs(os.path.dirname(filename))
    ply.write(filename)

def save_mesh(verts, faces, filename, colors=None):
    vertex = np.core.records.fromarrays(verts.transpose(), names='x, y, z', formats='f4, f4, f4')
    n = len(vertex)
    desc = vertex.dtype.descr

    if colors is not None:
        vertex_color = np.core.records.fromarrays(colors.transpose() * 255, names='red, green, blue',
                                                  formats='u1, u1, u1')
        assert len(vertex_color) == n
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(n, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    faces_building = []
    for i in range(0, faces.shape[0]):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = PlyElement.describe(vertex_all, "vertex")
    el_faces = PlyElement.describe(faces_tuple, "face")

    ply_data = PlyData([el_verts, el_faces])
    if not os.path.exists(os.path.dirname(filename)):
       os.makedirs(os.path.dirname(filename))
    ply_data.write(filename)

def sdf2mesh(sdf_grid, voxel_origin, step_size, mask=None):
    if (sdf_grid < 0).sum() == 0:
        return
    verts, faces, normals, values = skimage.measure.marching_cubes(
        sdf_grid, level=0.0, spacing=[step_size] * 3, mask=mask
    )

    verts = verts + voxel_origin

    return verts, faces