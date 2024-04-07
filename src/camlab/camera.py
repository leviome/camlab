import numpy as np

class CameraObj:
    def __init__(self, intri_mat=None, image_name=None):
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

    def load_extrinsic(self, extrinsic):
        self.R = extrinsic[:3, :3]
        self.T = extrinsic[:3, -1]

    def intri_check(self):
        if not self.is_intri_set:
            print("please set intri parameters !")
            raise Exception

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
        self.intri_check()
        return self.cam2screen(self.world2cam(p), to_int=to_int)

    def make_ray(self, screen_p, h=None, w=None, extri=None):
        self.intri_check()
        if h is None:
            h = self.h
        if w is None:
            w = self.w
        if extri is None:
            R = self.R
            T = self.T
        else:
            R = extri[:3, :3]
            T = extri[:3, -1]

        x, y = screen_p
        direction = np.array([(x - self.o_x) / self.focal_x, - (y - self.o_y) / self.focal_y, -1])
        ray_d = np.sum(direction[None] * R, -1)
        ray_o = T

        return ray_o, ray_d
     
    def quaternion2rotation(self, quater):
        self.touch += 1
        w, x, y, z = quater
        r11 = 1 - 2 * y * y - 2 * x * x
        r12 = 2 * x * y - 2 * z * w
        r13 = 2 * x * z + 2 * y * w
        r21 = 2 * x * y + 2 * z * w
        r22 = 1 - 2 * x * x - 2 * z * z
        r23 = 2 * y * z - 2 * x * w
        r31 = 2 * x * z - 2 * y * w
        r32 = 2 * y * z + 2 * x * w
        r33 = 1 - 2 * x * x - 2 * y * y

        rot = [[r11, r12, r13],
               [r21, r22, r23],
               [r31, r32, r33]]
        return np.array(rot)

    def colmap2extri(self, cmp):
        self.touch += 1
        quater = cmp[0:4]
        trans = cmp[4:]
        rot = self.quaternion2rotation(quater)
        extri = np.eye(4)
        extri[:3, :3] = rot
        extri[0][-1] = trans[0]
        extri[1][-1] = trans[1]
        extri[2][-1] = trans[2]
        return extri

       

def _test1():
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

def _test2():
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
    cam_obj.load_extrinsic(extri2)

    p = (400, 400)
    rayo, rayd = cam_obj.make_ray(p)
    ps = cam_obj.world2screen(rayd - rayo, to_int=True)
    print(rayo, rayd)
    print(ps)
    
    quater = [0.980317, 0.002613, -0.19732, 0.005964]
    print(cam_obj.quaternion2rotation(quater))

    colmap_params = [0.980317, 0.002613, -0.19732, 0.005964, 5.91824, 0.099605, 0.691238]
    print(cam_obj.colmap2extri(colmap_params))



if __name__ == "__main__":
    # _test1()
    _test2()
