import math
from .camera import *


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def load_3dgs_camera(gs_cam):
    """
    Support 3D Gaussian Splatting Camera-Class Object
    """
    cam = CameraObj()
    cam.manual_init(focal=fov2focal(cam_gs.FoVx, cam_gs.image_width),
                    w=cam_gs.image_width, h=cam_gs.image_height)

    ext = np.eye(4)
    ext[:3, :3] = np.transpose(cam_gs.R)
    ext[:3, 3] = cam_gs.T
    c2w = np.linalg.inv(ext)
    c2w[:3, 1:3] *= -1

    cam.R = c2w[:3, :3]
    cam.T = c2w[:3, 3]
    return cam
