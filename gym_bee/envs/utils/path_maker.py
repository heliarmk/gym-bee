import joblib
import numpy as np
import os

def compute_angle(vec1, vec2):
    """
    :param vec1: numpy one dimensional array
    :param vec2: numpy one dimensional array
    :return:     float angle [-pi ~ pi]
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    angle = (np.arctan2(vec1[1], vec1[0]) - np.arctan2(vec2[1], vec2[0])) % np.pi
    return angle / np.pi * 180.0


class path_file_maker:
    def __init__(self, type, point_num, direction, base_dir):
        """
        make path for 600x600 arena
        the bigest draw region is 480x480
        :param type: path type, e.g: point, square, circle, s-type
        :param point_num: the number for point type path
        :param direction clockwise or anti-clockwise for
        """
        self.radius = 240
        self.center = np.array([300, 300])
        self.path_file = {}
        self.type = type
        self.point_num = point_num
        self.direction = direction
        self.base_dir = base_dir

        if point_num <= 0:
            raise ValueError("The point num:%d is invalid" % point_num)
        if not direction in ["clockwise", "anti_clockwise"]:
            raise ValueError("The direction type is invalid, it must clockwise or anti_clockwise")

    def file_gen(self):
        self.path_file.update({"type":self.type})
        if self.type == "point":
            self.path_file.update({"init_pos": self.center})
            dir, path_list = self.path_list_maker(self.center, self.radius, self.point_num, self.direction)
            self.path_file.update({"init_dir": dir})
            self.path_file.update({"path_pos": path_list})
        elif self.type == "quad":
            self.polygon_path_maker(4)
        elif self.type == "hexo":
            self.polygon_path_maker(6)
        elif self.type == "oct":
            self.polygon_path_maker(8)
        elif self.type == "circle":
            self.polygon_path_maker(16)
        else:
            raise ValueError("The type %d is not implemented" % self.type)

        fname = os.path.join(self.base_dir, "path_file_type_" + self.type + ".pkl")
        joblib.dump(value=self.path_file, filename=fname, compress=3)

    def path_list_maker(self, center, radius, point_num, dir):
        path_point_list = []
        angle_step = 360.0 / point_num
        start_angle = np.random.randint(-180, 180, 1)
        angle_diff = np.array([radius * np.cos(np.deg2rad(start_angle)), radius * np.sin(np.deg2rad(start_angle))]).reshape(-1)
        start_point = center + angle_diff
        path_point_list.append({"pos": start_point})
        cur_angle = start_angle
        for i in range(1, point_num):
            if i == 1:
                cur_angle = float(start_angle)
            if dir == "clockwise":
                cur_angle -= angle_step
            elif dir == "anti_clockwise":
                cur_angle += angle_step
            else:
                raise ValueError("the dir %s is not accepted")

            if cur_angle > 180:
                cur_angle = -360 + cur_angle
            if cur_angle < -180:
                cur_angle = 360 + cur_angle
            point = center + (radius * np.cos(np.deg2rad(cur_angle)), radius * np.sin(np.deg2rad(cur_angle)))
            path_point_list.append({"pos": point})

        return start_angle, path_point_list

    def polygon_path_maker(self, side_num):

        dir, path_list = self.path_list_maker(self.center, self.radius, side_num, self.direction)
        first_point = np.asarray(path_list[0]["pos"])
        second_point = np.asarray(path_list[1]["pos"])
        self.path_file.update({"init_pos": first_point})
        self.path_file.update({"init_dir": compute_angle(second_point - first_point, (1, 0))})
        self.path_file.update({"path_pos": path_list})

if __name__ == "__main__":
    maker = path_file_maker("hexo", 8, "clockwise", "../path_file")
    maker.file_gen()