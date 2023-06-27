import numpy as np

def Rx(angle):
    cosval, sinval = np.cos(angle), np.sin(angle)
    rotate_mat = np.eye(3)
    rotate_mat[1][1] = cosval
    rotate_mat[2][2] = cosval
    rotate_mat[1][2] = -sinval
    rotate_mat[2][1] = sinval
    return rotate_mat

def Ry(angle):
    cosval, sinval = np.cos(angle), np.sin(angle)
    rotate_mat = np.eye(3)
    rotate_mat[0][0] = cosval
    rotate_mat[2][2] = cosval
    rotate_mat[0][2] = sinval
    rotate_mat[2][0] = -sinval
    return rotate_mat

def Rz(angle):
    cosval, sinval = np.cos(angle), np.sin(angle)
    rotate_mat = np.eye(3)
    rotate_mat[0][0] = cosval
    rotate_mat[1][1] = cosval
    rotate_mat[0][1] = -sinval
    rotate_mat[1][0] = sinval
    return rotate_mat

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, surface, nonface, normals, sdfs):
        for t in self.transforms:
            surface, nonface, normals, sdfs = t(surface, nonface, normals, sdfs)
        return surface, nonface, normals, sdfs

class PointCloudRotate(object):
    def __init__(self, angle_sigma=0.18, angle_clip=0.54):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )
        return angles

    def __call__(self, surface, nonface, normals, sdfs):
        angles = self._get_angles()
        rx = Rx(angles[0])
        ry = Ry(angles[1])
        rz = Rz(angles[2])

        rotation_matrix = np.matmul(np.matmul(rz, ry), rx)
        if normals is None:
            surface = np.matmul(surface, rotation_matrix.T)
            nonface = np.matmul(nonface, rotation_matrix.T)
            return surface, nonface, normals, sdfs
        else:
            surface = np.matmul(surface, rotation_matrix.T)
            nonface = np.matmul(nonface, rotation_matrix.T)
            normals = np.matmul(normals, rotation_matrix.T)

            return surface, nonface, normals, sdfs

class PointCloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, surface, nonface, normals, sdfs):
        scaler = np.random.uniform(self.lo, self.hi)
        surface = surface * scaler
        nonface = nonface * scaler
        sdfs = sdfs * scaler

        return surface, nonface, normals, sdfs

class PointCloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, surface, nonface, normals, sdfs):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        surface = surface + translation
        nonface = nonface + translation
        return surface, nonface, normals, sdfs


class PointCloudJitter(object):
    def __init__(self, jit=0.005):
        self.jit = jit

    def __call__(self, surface, nonface, normals, sdfs):
        surface = surface + 0.005 * np.random.randn(surface.shape[0], 3)
        return surface, nonface, normals, sdfs