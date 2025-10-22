import torch
import numpy as np


class CameraObjTensor:
    def __init__(self, intri_mat=None,
                 image_name=None,
                 device="cuda:0"):
        self.device = device

        # intrinsic
        self.focal_x = None
        self.focal_y = None
        self.o_x = None
        self.o_y = None
        self.K = None
        self.intrinsic = torch.eye(3)
        if intri_mat is not None:
            self.intrinsic = torch.FloatTensor(intri_mat).to(self.device)
            self.intrinsic[0][0] *= -1

        # screen
        self.h = None
        self.w = None

        # extrinsic
        self.R = None
        self.Tr = None
        self.is_intri_set = False

        # image
        self.image_name = image_name

        self.touch = 0

        if intri_mat:
            self.K = intri_mat
            self.focal_x = intri_mat[0][0]
            self.focal_y = intri_mat[1][1]
            self.o_x = intri_mat[0][2]
            self.o_y = intri_mat[1][2]
            self.w = int(2 * self.o_x + .5)
            self.h = int(2 * self.o_y + .5)
            self.is_intri_set = True

    def manual_init(self, focal, w, h):
        if isinstance(focal, list):
            self.focal_x = focal[0]
            self.focal_y = focal[1]
        else:
            self.focal_x = focal
            self.focal_y = focal
        self.w = w
        self.h = h
        self.o_x = w / 2
        self.o_y = h / 2
        self.is_intri_set = True

        intrinsic = np.eye(3)
        intrinsic[0][0] = self.focal_x * -1
        intrinsic[1][1] = self.focal_y
        intrinsic[0][2] = self.o_x
        intrinsic[1][2] = self.o_y

        self.intrinsic = torch.FloatTensor(intrinsic)
        self.intrinsic = self.intrinsic.to(self.device)

    def load_extrinsic(self, extrinsic):
        self.R = torch.FloatTensor(extrinsic[:3, :3]).to(self.device)
        self.Tr = torch.FloatTensor(extrinsic[:3, -1]).to(self.device)

    def intri_check(self):
        if not self.is_intri_set:
            print("please set intri parameters !")
            raise Exception

    def screen2cam(self, p, depth):
        x, y = p
        pz = torch.tensor([x, y, 1.0], dtype=torch.float32).to(self.device)
        K_inv = torch.inverse(self.intrinsic)
        return depth * torch.matmul(K_inv, pz)
   
    def screen2cam_batch(self, p, depth):
        """
        p: [N, 2] (may be int); depth: [N]
        Returns: cam_coords [N, 3]
        """
        # ensure inputs are float and on the camera device
        p = p.to(self.device, dtype=torch.float32)
        depth = depth.to(self.device, dtype=torch.float32)

        x = p[:, 0]
        y = p[:, 1]
        ones = torch.ones_like(x, dtype=torch.float32, device=self.device)
        pz = torch.stack([x, y, ones], dim=1)  # [N, 3]

        K_inv = torch.inverse(self.intrinsic)  # [3, 3]
        cam_coords = torch.matmul(K_inv, pz.T).T  # [N, 3]
        cam_coords = cam_coords * depth[:, None]  # [N, 3]
        return cam_coords  # [N, 3]

    def cam2world(self, p):
        if self.R is None or self.Tr is None:
            print("Please init the extrinsic params!")
            return -1
        return torch.matmul(p, self.R.T) + self.Tr

    def world2cam(self, p):
        if self.R is None or self.Tr is None:
            print("Please init the extrinsic params!")
            return -1
        return torch.matmul(p - self.Tr, self.R).T

    def cam2world_batch(self, p):
        if self.R is None or self.Tr is None:
            print("Please init the extrinsic params!")
            return -1
        return torch.matmul(p, self.R.T) + self.Tr.unsqueeze(0)

    def cam2screen(self, p):
        """
                          x                                y
            x_im = f_x * --- + offset_x      y_im = f_y * --- + offset_y
                          z                                z
        """
        res = torch.matmul(self.intrinsic, p) / p[2]
        depth = p[2] # - self.focal_x  # Suppose focal_x = focal_y
        return res.T[:, :2], depth

    def world2screen(self, p, to_int=False):
        self.intri_check()
        res, depth = self.cam2screen(self.world2cam(p))
        if to_int:
            res = res.long()
        return res, depth

    def prune_gaussians_by_box(self, gaussians, box):
        x1, x2, y1, y2 = box
        p, _ = self.world2screen(gaussians._xyz, to_int=True)
        in_indices = (p[:, 0] >= x1) & (p[:, 0] < x2) & (p[:, 1] >= y1) & (p[:, 1] < y2)
        return in_indices


    def prune_gaussians_by_mask(self, gaussians, mask, keep_out_screen=False):
        p, _ = self.world2screen(gaussians._xyz, to_int=True)
        h, w = mask.shape
        in_screen = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        out_screen = ~ in_screen
        flag = torch.zeros_like(in_screen)
        p_in = p[in_screen]
        flag[in_screen] = mask[p_in[:, 1], p_in[:, 0]]
        if keep_out_screen:
            flag = flag | out_screen
        return flag


def _test():
    intri = [[1111.0, 0.0, 400.0],
             [0.0, 1111.0, 400.0],
             [0.0, 0.0, 1.0]]
    extri = np.array(
        [[-9.9990e-01, 4.1922e-03, -1.3346e-02, -5.3798e-02],
         [-1.3989e-02, -2.9966e-01, 9.5394e-01, 3.8455e+00],
         [-4.6566e-10, 9.5404e-01, 2.9969e-01, 1.2081e+00],
         [0.0, 0.0, 0.0, 1.0]])

    cam_obj = CameraObjTensor(intri)
    cam_obj.load_extrinsic(extri)

    a = [[-1, 1, -1] for _ in range(1000000)]
    p = torch.FloatTensor(a)
    p = p.to("cuda:0")
    p1 = cam_obj.world2screen(p)
    print(p1[0])
    print(p1.shape)


if __name__ == "__main__":
    _test()

