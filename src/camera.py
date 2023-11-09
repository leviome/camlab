import numpy as np

class CameraObj:
    def __init__(self, intri_mat=None):
        # intrinsic
        self.focal_x = None
        self.focal_y = None
        self.o_x = None
        self.o_y = None
        self.K = None

        # screen
        self.h = None
        self.w = None

        # extrinsic
        self.R = None
        self.T = None

        if intri_mat:
            self.K = intri_mat
            self.focal_x = intri_mat[0][0]
            self.focal_y = intri_mat[1][1]
            self.o_x = intri_mat[0][2]
            self.o_y = intri_mat[1][2]
            self.w = int(2 * self.o_x + .5)
            self.h = int(2 * self.o_y + .5)

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

    def load_extrinsic(self, extrinsic):
        self.R = extrinsic[:3, :3]
        self.T = extrinsic[:3, -1]

    def world2cam(self, p):
        if self.R is None or self.T is None:
            print("Please init the extrinsic params!")
            return -1
        return np.dot(p - self.T, self.R)

    def cam2screen(self, p, to_int=False):
        """
                          x                                y
            x_im = f_x * --- + offset_x      y_im = f_y * --- + offset_y
                          z                                z
        """
        x, y, z = p
        if not to_int:
            return [-x * self.focal_x / z + self.o_x, y * self.focal_y / z + self.o_y]
        else:
            return [int(-x * self.focal_x / z + self.o_x + .5), int(y * self.focal_y / z + self.o_y + .5)]

    def world2screen(self, p, to_int=False):
        return self.cam2screen(self.world2cam(p), to_int=to_int)
       

def _test_cam():
    import cv2
    intri = [[1111.0, 0.0, 400.0],
         [0.0, 1111.0, 400.0],
         [0.0, 0.0, 1.0]]
    extri = np.array(
        [[-9.9990e-01,  4.1922e-03, -1.3346e-02, -5.3798e-02],
        [-1.3989e-02, -2.9966e-01,  9.5394e-01,  3.8455e+00],
        [-4.6566e-10,  9.5404e-01,  2.9969e-01,  1.2081e+00],
        [0.0, 0.0, 0.0, 1.0]])
    extri1 = np.array([[-3.0373e-01, -8.6047e-01,  4.0907e-01,  1.6490e+00],
        [ 9.5276e-01, -2.7431e-01,  1.3041e-01,  5.2569e-01],
        [-7.4506e-09,  4.2936e-01,  9.0313e-01,  3.6407e+00],
        [0.0, 0.0, 0.0, 1.0]])
    extri2 = np.array(
        [[0.4429636299610138, 0.31377720832824707, -0.8398374915122986, -3.385493516921997],
         [-0.8965396881103516, 0.1550314873456955, -0.41494810581207275, -1.6727094650268555],
         [0.0, 0.936754584312439, 0.3499869406223297, 1.4108426570892334],
         [0.0, 0.0, 0.0, 1.0]])

    cam_obj = CameraObj(intri)
    cam_obj.load_extrinsic(extri)

    vertex = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1]]
    # vertex = [[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5], [-.5, .5, .5], [.5, .5, .5], [.5, -.5, .5], [-.5, -.5, .5]]

    # blank = np.zeros((800, 800, 3), np.uint8)
    blank = cv2.imread("r_0.png")
    print(blank.shape)

    p_s = [cam_obj.world2screen(p_, to_int=True) for p_ in vertex]
    print(p_s)

    for i in range(len(p_s)):
        p1 = p_s[i]
        p2 = p_s[i + 1] if i < len(p_s) - 1 else p_s[0]
        cv2.line(blank, p1, p2, (255, 255, 0), 1)
    cv2.line(blank, p_s[3], p_s[0], (255, 255, 0), 1)
    cv2.line(blank, p_s[4], p_s[7], (255, 255, 0), 1)
    cv2.line(blank, p_s[1], p_s[6], (255, 255, 0), 1)
    cv2.line(blank, p_s[2], p_s[5], (255, 255, 0), 1)
    cv2.imwrite("demo.png", blank)


if __name__ == "__main__":
    _test_cam()
