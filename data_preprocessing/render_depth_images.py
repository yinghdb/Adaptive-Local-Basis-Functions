"""Renders depth scans of an input mesh for the depth completion task."""
import os
import sys

import numpy as np
import pyrender
import trimesh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--class_ids', type=str, default="")
parser.add_argument('--depth_root_dir', type=str, default="")
parser.add_argument('--data_build_dir', type=str, default="")
parser.add_argument('--n_images_per_mesh', type=int, default=16)
parser.add_argument('--height', type=int, default=224)
parser.add_argument('--width', type=int, default=224)
parser.add_argument('--yfov', type=float, default=np.pi / 3.0)

args = parser.parse_args()

# The context is OpenGL- it's global and not threadsafe.
# Note that this context still requires a headed server even those it's
# offscreen. If you need a headless one, you can switch to egl/osmesa with
# pyrender via environment variables, but those can be trickier to setup.
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def look_at_np(eye, center, world_up):
    """Computes camera viewing matrices.

    Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).

    Args: 
    eye: 2-D float32 numpy array (or convertible value) with shape
        [batch_size, 3] containing the XYZ world space position of the camera.
    center: 2-D float32 array (or convertible value) with shape [batch_size, 3]
        containing a position along the center of the camera's gaze.
    world_up: 2-D float32 array (or convertible value) with shape [batch_size,
        3] specifying the world's up direction; the output camera will have no
        tilt with respect to this direction.

    Returns:
    A [batch_size, 4, 4] numpy array containing a right-handed camera
    extrinsics matrix that maps points from world space to points in eye space.
    """
    batch_size = center.shape[0]

    vector_degeneracy_cutoff = 1e-6
    forward = center - eye
    forward_norm = np.linalg.norm(forward, ord=2, axis=1, keepdims=True)
    assert forward_norm >= vector_degeneracy_cutoff
    forward = np.divide(forward, forward_norm)

    to_side = np.cross(forward, world_up)
    to_side_norm = np.linalg.norm(to_side, ord=2, axis=1, keepdims=True)
    assert to_side_norm >= vector_degeneracy_cutoff
    to_side = np.divide(to_side, to_side_norm)
    cam_up = np.cross(to_side, forward)

    w_column = np.array(
        batch_size * [[0., 0., 0., 1.]], dtype=np.float32)  # [batch_size, 4]
    w_column = np.reshape(w_column, [batch_size, 4, 1])
    view_rotation = np.stack(
        [to_side, cam_up, -forward,
        np.zeros_like(to_side, dtype=np.float32)],
        axis=1)  # [batch_size, 4, 3] matrix
    view_rotation = np.concatenate([view_rotation, w_column],
                                    axis=2)  # [batch_size, 4, 4]
    identity_singleton = np.eye(3, dtype=np.float32)[np.newaxis, ...]
    identity_batch = np.tile(identity_singleton, [batch_size, 1, 1])
    view_translation = np.concatenate([identity_batch, np.expand_dims(-eye, 2)], 2)
    view_translation = np.concatenate(
        [view_translation,
        np.reshape(w_column, [batch_size, 1, 4])], 1)
    camera_matrices = np.matmul(view_rotation, view_translation)
    return camera_matrices

def render_depth_image(mesh, cam2world, context):
    scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_node = pyrender.Node(mesh=pyr_mesh, matrix=np.eye(4))
    scene.add_node(mesh_node)
    ar = args.width / float(args.height)
    cam = pyrender.PerspectiveCamera(yfov=args.yfov, aspectRatio=ar)
    scene.add_node(pyrender.Node(camera=cam, matrix=cam2world))
    color, depth = context.render(scene)
    return depth


def clips_edge(depth):
    assert depth.shape[0] == args.height
    assert depth.shape[1] == args.width
    top_row = depth[0, :]
    bottom_row = depth[args.height-1, :]
    left_col = depth[:, 0]
    right_col = depth[:, args.width-1]
    edges = np.concatenate([top_row, bottom_row, left_col, right_col])
    return np.any(edges)


def get_cam2world(center, eye, world_up):
    eye = np.reshape(eye, [1, 3])
    world_up = np.reshape(world_up, [1, 3])
    center = np.reshape(center, [1, 3])
    world2cam = look_at_np(eye = eye, center=center,
        world_up=world_up)
    return np.linalg.inv(world2cam[0, ...])
  

def find_critical_radius(mesh, dir_to_eye, center, world_up, context,
    min_radius=1.0, max_radius=5.0, iterations=5, fallback_radius=1.5):
    def radius_clips(radius):
        cam2world = get_cam2world(eye=dir_to_eye*radius, center=center,
            world_up=world_up)
        depth = render_depth_image(mesh, cam2world, context)
        return clips_edge(depth)
    if radius_clips(max_radius):
        return max_radius
    if not radius_clips(min_radius):
        return min_radius
    for i in range(iterations):
        midpoint = (min_radius + max_radius) / 2.0
        if radius_clips(midpoint):
            min_radius = midpoint
        else:
            max_radius = midpoint
    return max_radius

def find_radius(mesh, dir_to_eye, center, world_up, context,
    start_radius, step_radius=0.2, iterations=10):
    def radius_clips(radius):
        cam2world = get_cam2world(eye=dir_to_eye*radius, center=center,
            world_up=world_up)
        depth = render_depth_image(mesh, cam2world, context)
        return clips_edge(depth)
    cur_radius = start_radius
    for i in range(iterations):
        if not radius_clips(cur_radius):
            return cur_radius
        cur_radius += step_radius
    return cur_radius

def sample_depth_image(mesh, context, idx, step_size):
    center = np.random.randn(3).astype(np.float32) * 0.05
    world_up = np.array([0, 1, 0], dtype=np.float32)

    angle_rand = np.random.rand(3)
    y_rot = idx * step_size + angle_rand[0] * step_size - step_size / 2.0
    x_rot = 10 + angle_rand[1] * 50
    theta = np.deg2rad(y_rot)
    phi = np.deg2rad(x_rot)
    camY = np.sin(phi)
    temp = np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)

    dir_to_eye = np.array([camX, camY, camZ])
    start_radius = 1.0 + angle_rand[2] * 0.5
    radius = find_radius(mesh, dir_to_eye, center, world_up, context, start_radius)

    eye = dir_to_eye * radius
    cam2world = get_cam2world(eye=eye, center=center, world_up=world_up)
    return render_depth_image(mesh, cam2world, context), cam2world

def sample_depth_images(mesh, context, n_images=16):
    images = []
    cam2worlds = []
    step_size = 360.0 / n_images
    for i in range(n_images):
        depth, cam2world = sample_depth_image(mesh, context, i, step_size)
        images.append(depth)
        cam2worlds.append(cam2world)
    return np.stack(images), np.stack(cam2worlds)


def get_projection_matrix():
    width = float(args.width)
    height = float(args.height)
    aspect_ratio = width/height
    fov = args.yfov
    K = np.zeros((3,3), dtype=np.float32)
    K[0][0] = width / 2 / np.tan(fov / 2)
    K[1][1] = -height / 2. / np.tan(fov / 2) * aspect_ratio
    K[0][2] = -(width-1) / 2.
    K[1][2] = -(height-1) / 2.
    K[2][2] = -1.

    return K


if __name__ == "__main__":
    class_ids = args.class_ids.split(',')
    context = pyrender.OffscreenRenderer(viewport_width=args.width, viewport_height=args.height, point_size=1.0)
    # data list 
    for class_id in class_ids:
        class_dir = os.path.join(args.data_build_dir, class_id, "4_watertight_scaled")
        for obj_id in os.listdir(class_dir):
            obj_id = obj_id.split('.')[0]
            input_mesh = os.path.join(class_dir, obj_id+".off")
            output_dir = os.path.join(args.depth_root_dir, class_id, obj_id)
            
            print(f"Rendering depth image {input_mesh}")
            if not os.path.exists(output_dir):
                mesh = trimesh.load(input_mesh)
                n_images_per_mesh = args.n_images_per_mesh
                images, cam2world = sample_depth_images(mesh, context, n_images_per_mesh)
                projection_matrix = get_projection_matrix()

                os.makedirs(output_dir, exist_ok=True)
                for i in range(n_images_per_mesh):
                    output_npz = os.path.join(output_dir, "%02d.npz" % (i))
                    np.savez(output_npz, depth=images[i], cam2world=cam2world[i], projection=projection_matrix)



