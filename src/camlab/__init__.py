import math
import torch
from .camera import *
from .camera_torch import *


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def load_3dgs_camera(cam_gs, device="cpu"):
    """
    Support 3D Gaussian Splatting Camera-Class Object
    """
    if device.startswith("cpu"):
        cam = CameraObj(image_name=cam_gs.image_name)
        cam.manual_init(focal=fov2focal(cam_gs.FoVx, cam_gs.image_width),
                        w=cam_gs.image_width, h=cam_gs.image_height)

        ext = np.eye(4)
        ext[:3, :3] = np.transpose(cam_gs.R)
        ext[:3, 3] = cam_gs.T
        c2w = np.linalg.inv(ext)
        c2w[:3, 1:3] *= -1

        cam.R = c2w[:3, :3]
        cam.T = c2w[:3, 3]
    else:
        cam = CameraObjTensor(image_name=cam_gs.image_name,
                              device=device)
        cam.manual_init(focal=fov2focal(cam_gs.FoVx, cam_gs.image_width),
                        w=cam_gs.image_width, h=cam_gs.image_height)
        ext = np.eye(4)
        ext[:3, :3] = np.transpose(cam_gs.R)
        ext[:3, 3] = cam_gs.T
        c2w = np.linalg.inv(ext)
        c2w[:3, 1:3] *= -1

        cam.R = torch.FloatTensor(c2w[:3, :3]).to(device)
        cam.Tr = torch.FloatTensor(c2w[:3, 3]).to(device)

    return cam


def approximate(n, o=1e8):
    return int(n * o) / o


def closest_points_between_lines(line1, line2, acc=8):
    # Convert points to numpy arrays for easier manipulation
    p1, p2 = np.array(line1)
    q1, q2 = np.array(line2)

    # Direction vectors for each line
    v1 = p2 - p1
    v2 = q2 - q1

    # Define matrices A and B for the equations
    A = np.vstack((v1, -v2)).T
    B = q1 - p1

    # Solve the equation A * [t, s] = B
    t, s = np.linalg.lstsq(A, B, rcond=None)[0]

    # Calculate closest points on each line
    closest_point_line1 = p1 + t * v1
    closest_point_line2 = q1 + s * v2

    # Calculate the distance
    expo = pow(10, acc)
    distance = np.linalg.norm(closest_point_line1 - closest_point_line2, ord=2)
    distance = approximate(distance, o=expo)
    closest_point_line1 = [approximate(n, o=expo) for n in list(closest_point_line1)]
    closest_point_line2 = [approximate(n, o=expo) for n in list(closest_point_line2)]

    return closest_point_line1, closest_point_line2, distance

