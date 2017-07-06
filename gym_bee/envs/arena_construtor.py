#!/usr/bin/env python
# coding=utf-8

# TODO 添加构建实验场地的方法类，包括不同类型的场地，手动和自动生成目标路径点的方法等

import numpy as np
from .RW import boundary_condition
#matplotlib
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


class ManualPathBuilder:
    def __init__(self, rw_domain, line, path, path_point_num, detect_radius, length_range):
        """
        :param rw_domain:
        :param line:
        :param path_point:
        :param point_num:
        :param detect_radius:
        :param length_range: the range of the length of adjacent point in the path
        :type two dim np.array, like [1,10]
        """
        self.domain = rw_domain
        self.line = line
        self.path = path
        self.point_num = path_point_num
        self.detect_radius = detect_radius
        self.length_range = length_range
        self.boundaries = []
        self.xs = list(self.line.get_xdata())
        self.ys = list(self.line.get_ydata())
        self.cid = self.line.figure.canvas.mpl_connect('button_press_event', self)
        self.set_boundary()

    def __call__(self, event):
        print('click', event)
        # check in the axes
        if event.inaxes != self.line.axes: return
        cur_coor = np.array([event.xdata, event.ydata])
        # check if num of point in path is exceed setting
        if len(self.path) >= self.point_num:
            self.line.figure.canvas.mpl_disconnect(self.cid)
            plt.close("all")
            return
        # check if hit boundaries
        if self.check_hit_boundary(self.path[-1], cur_coor):
            print("hit boundary")
            return
        # check if overlap
        if self.check_overlap(cur_coor):
            print("overlap")
            return
        # check if out of range
        if not self.check_length_in_range(self.path[-1], cur_coor):
            print("length not in range")
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.path.append(cur_coor)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

    def set_boundary(self):
        for i, edge in enumerate(self.domain.edges):
            newBc = self.gen_boundary_condition(i, edge)
            self.boundaries.append(newBc)

    def gen_boundary_condition(self, BcId, edge):
        if BcId == None:
            Id = -1
        else:
            Id = BcId
        return boundary_condition.setBack(self, Id, edge, direcBehaviour=0)

    def check_hit_boundary(self, pre_coor, cur_coor):
        """
        :param pre_coor: the coordinate of the previous point in the path
        :type np.array
        :param cur_coor: the coordinate of current point in the path
        :type np.array
        :return: True if hit boundary otherwise False
        """
        if self.domain.typ == "circle":
            after_check_coor = self.boundaries[0].hit(pre_coor, cur_coor)
            if not (after_check_coor == cur_coor).all():
                print("hit boundary")
                return True
            else:
                return False
        else:
            for boundary in self.boundaries:
                after_check_coor = boundary.hit(pre_coor, cur_coor)

                if not (after_check_coor == cur_coor).all():
                    print("hit boundary")
                    return True

            return False

    def check_overlap(self, cur_coor):
        """
        :param cur_coor: the current point's coordinate of the path
        :return: True if the overlap otherwise False
        """
        for point in self.path:
            if np.linalg.norm(cur_coor - point) <= 2 * self.detect_radius:
                return True
        else:
            return False

    def check_length_in_range(self, pre_coor, cur_coor):
        """
        :param pre_coor: the coordinate of the previous point in the path
        :type np.array
        :param cur_coor: the coordinate of current point in the path
        :type np.array
        :return: True if the length of vector(cur_coor - pre_coor) in the of length range otherwise return false
        """
        vec_norm = np.linalg.norm(cur_coor - pre_coor)
        if vec_norm < self.length_range[0] or vec_norm > self.length_range[1]:
            print(vec_norm)
            return False
        else:
            return True


class AutoPathBuilder:
    def __init__(self, rw_domain, path, path_point_num, detect_radius, length_range):
        self.domain = rw_domain
        self.path = path
        self.point_num = path_point_num
        self.detect_radius = detect_radius
        self.length_range = length_range
        self.boundaries = []
        self.set_boundary()
        self.set_path_point()

    def set_path_point(self):
        while len(self.path) < self.point_num:
            pre_coor = self.path[-1]
            rand_dist = np.random.uniform(self.length_range[0], self.length_range[1])
            rand_dire = np.random.uniform(-1, 1) * np.pi
            cur_coor = pre_coor + np.array([rand_dist * np.cos(rand_dire), rand_dist * np.sin(rand_dire)])
            # print(pre_coor, cur_coor)
            # check if hit boundaries
            if self.check_hit_boundary(pre_coor, cur_coor):
                #print("hit boundary\n")
                continue
            # check if overlap
            if self.check_overlap(cur_coor):
                #print("overlap\n")
                continue
            self.path.append(cur_coor)
        return

    def set_boundary(self):
        for i, edge in enumerate(self.domain.edges):
            newBc = self.gen_boundary_condition(i, edge)
            self.boundaries.append(newBc)

    def gen_boundary_condition(self, BcId, edge):
        if BcId == None:
            Id = -1
        else:
            Id = BcId
        return boundary_condition.setBack(self, Id, edge, direcBehaviour=0)

    def check_hit_boundary(self, pre_coor, cur_coor):
        """
        :param pre_coor: the coordinate of the previous point in the path
        :type np.array
        :param cur_coor: the coordinate of current point in the path
        :type np.array
        :return: True if hit boundary otherwise False
        """
        if self.domain.typ == "circle":
            after_check_coor = self.boundaries[0].hit(pre_coor, cur_coor)
            if not (after_check_coor == cur_coor).all():
                return True
            else:
                return False
        else:
            for boundary in self.boundaries:
                _, ret = boundary.hit(pre_coor, cur_coor)

                if ret:
                    return True

            return False

    def check_overlap(self, cur_coor):
        """
        :param cur_coor: the current point's coordinate of the path
        :return: True if the overlap otherwise False
        """
        for point in self.path:
            if np.linalg.norm(cur_coor - point) <= 2 * self.detect_radius:
                return True
        else:
            return False

    def check_length_in_range(self, pre_coor, cur_coor):
        """
        :param pre_coor: the coordinate of the previous point in the path
        :type np.array
        :param cur_coor: the coordinate of current point in the path
        :type np.array
        :return: True if the length of vector(cur_coor - pre_coor) in the of length range otherwise return false
        """
        vec_norm = np.linalg.norm(cur_coor, pre_coor)
        if vec_norm < self.length_range[0] or vec_norm > self.length_range[1]:
            return False
        else:
            return True


class ArenaConstru(object):
    def __init__(self, rw_domain, path, detect_radius, length_range, route_plan_way="Manual", path_point_num=5):
        """

        :param rw_domain: random walk domain instance
        """
        self.domain = rw_domain
        self.path = path
        self.env = self.domain.env
        self.detect_radius = detect_radius
        self.length_range = length_range
        self.route_plan_way = route_plan_way
        self.path_point_num = path_point_num
        self.initial_coordinate = self.env.initial_coordinate
        self.x_limit = self.env.x_limit
        self.y_limit = self.env.y_limit

    def draw_route(self):
        self.domain.draw()
        if not hasattr(self.env, 'fig_traj'):
            self.env.fig_traj = plt.figure()
            self.env.ax_traj = self.env.fig_traj.add_subplot(111)

        self.env.ax_traj.set_title('click to build line segments')
        # self.env.ax_traj.set_xlim([0, self.x_limit])
        # self.env.ax_traj.set_ylim([0, self.y_limit])
        line, = self.env.ax_traj.plot(self.initial_coordinate[0], self.initial_coordinate[1],
                                      color="#1F4E79", linestyle="solid", linewidth=2, marker="o",
                                      markerfacecolor="#5B9BD5", markersize=12,
                                      markeredgecolor="#BDD7EE", markeredgewidth=8)  # empty line

        ManualPathBuilder(self.domain, line, self.path, self.path_point_num, self.detect_radius, self.length_range)
        plt.show()

    def gen_route(self):
        self.path.append(np.array(self.initial_coordinate))
        if self.route_plan_way == "Auto":
            AutoPathBuilder(self.domain, self.path, self.path_point_num, self.detect_radius, self.length_range)
        elif self.route_plan_way == "Manual":
            self.draw_route()

    def construct_rect(self, vertex_offset, len_x, len_y):
        """
        :param vertex_offset: offset of the vertex of rect which has the low coordinate
        :type vertex_offset: np.array([axis_1, axis_2])
        :param len_x: side length in x direction
        :type len_x: float
        :param len_y: side length in y direction
        :type len_y: float
        :return: pyrw.geometry.rectangle -- rectangle object
        """
        v_offset = self.domain.addVertex(vertex_offset)
        self.domain.addRectangle(v_offset, len_x, len_y)
        if len(self.path) == 0:
            self.gen_route()
            # def construct_maze(self, maze_type):
            #    if maze_type == "y_maze":




            #    else:
            #        return
