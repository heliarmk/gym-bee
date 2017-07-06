from .utils.vec2d import vec2d
from .utils import percent_round_int
from .utils.multi_discrete import DiscreteToMultiDiscrete

# gym
import gym
from gym import error, spaces, utils

# numpy
import numpy as np
# scipy
from scipy.stats import skewnorm, vonmises_line, norm


class Target():
    def __init__(self, idx, pos_init, radius, color, SCREEN_WIDTH, SCREEN_HEIGHT):
        # super(Target, self).__init__()

        self.pos = vec2d(pos_init)
        self.color = color
        self.draw_color = tuple([int(x * 255) for x in self.color])

        self.t_index = idx
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.radius = radius
        self.rect = (2 * radius, 2 * radius)
        l = percent_round_int(-radius, 1.0)
        r = percent_round_int(radius, 1.0)
        t = percent_round_int(2 * radius, 1.0)
        b = 0
        self.rect_v = [(l, b), (l, t), (r, t), (r, b)]
        self.center = (pos_init[0] - radius, pos_init[1] - radius)
        '''
        image = pygame.Surface((2 * radius, 2 * radius))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        
        pygame.draw.circle(
            image,
            color,
            (radius, radius),
            radius,
            0
        )
        
        #target is a rect
        pygame.draw.rect(
            image,
            color,
            (0, 0, 2*radius, 2*radius)
        )
        if not self.t_index == None:
            self.text = ptext.draw(text=str(self.t_index), centery=True, center=(radius, radius),
                                   fontsize=percent_round_int(radius, 2.0), surf=image)
        '''

    def render(self, viewer):
        from gym.envs.classic_control import rendering
        self.image = rendering.make_polygon(self.rect_v, True)
        self.image.add_attr(rendering.Transform(translation=self.center))
        self.image.set_color(self.color[0], self.color[1], self.color[2])
        viewer.add_onetime(self.image)

    def draw(self, arr):
        from PIL import ImageDraw
        self.canvas = ImageDraw.Draw(arr)
        boundary = (
        percent_round_int(self.center[0] - self.radius, 1.0), percent_round_int(self.center[1] - self.radius, 1.0),
        percent_round_int(self.center[0] + self.radius, 1.0), percent_round_int(self.center[1] + self.radius, 1.0))
        self.canvas.rectangle(xy=boundary, fill=self.draw_color)


class BeePlayer():
    def __init__(self, dir, pos_init, radius, color, SCREEN_WIDTH, SCREEN_HEIGHT, ACTION_LIMIT):
        # super(BeePlayer, self).__init__()

        self.step_len = None
        self.dir = dir
        self.pos = vec2d(pos_init)
        self.pre_pos = vec2d(pos_init)
        self.color = color
        self.draw_color = tuple([int(x * 255) for x in self.color])
        self.radius = radius
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.ACTION_LIMIT = ACTION_LIMIT

        '''
        image = pygame.Surface((2 * radius, 2 * radius))
        image.fill((0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.circle(
            image,
            color,
            (radius, radius),
            radius,
            0
        )
        '''

        self.image = None
        self.center = pos_init

        # step_len skew-norm params

        # these value is measure in 1500px x 1500px
        self.loc = 1.4605 / 6.25
        self.scale = 1.8038
        self.shape = 3.5536
        """
        these value is measure in 100px x 100px
        self.loc = 0.160
        self.scale = 0.076
        self.shape = 1.129
        
        # these value is measure in 400px x 400px
        self.loc = 0.641
        self.scale = 0.305
        self.shape = 1.129
        """
        self.angle_loc = 0.002619
        self.angle_shape = 5.066
        self.angle_scale = 0.4992
        # angle diff norm param
        # self.angle_loc = 0.0025229
        # self.angle_shape = 1.8923
        # self.angle_scale = 0.49916
        # norm fit angle diff
        self.angle_loc_norm = -0.0001853
        self.angle_scale_norm = 0.2446

        self.angle_diff_min = -0.5 * np.pi
        self.angle_diff_max = 0.5 * np.pi

        self.dc_ul, self.dc_ll, self.fre_ul, self.fre_ul, self.fre_ll, self.pc_ul, self.pc_ll, self.ss_ul, self.ss_ll = ACTION_LIMIT

    def update(self, action):
        self.pre_pos.x = self.pos.x
        self.pre_pos.y = self.pos.y
        step_len = np.abs(skewnorm.rvs(a=self.shape, loc=self.loc, scale=self.scale, size=(1)))

        self._forward_dynamics(action)

        self.pos.x = n_x = self.pos.x + step_len * np.cos(self.dir)
        self.pos.y = n_y = self.pos.y + step_len * np.sin(self.dir)

        self.center = (n_x, n_y)

    def render(self, viewer):
        from gym.envs.classic_control import rendering
        self.image = rendering.make_circle(self.radius)
        self.image.add_attr(rendering.Transform(translation=self.center))
        self.image.set_color(self.color[0], self.color[1], self.color[2])
        viewer.add_onetime(self.image)

    def draw(self, arr):
        from PIL import ImageDraw
        self.canvas = ImageDraw.Draw(arr)
        boundary = (
            percent_round_int(self.center[0] - self.radius, 1.0), percent_round_int(self.center[1] - self.radius, 1.0),
            percent_round_int(self.center[0] + self.radius, 1.0), percent_round_int(self.center[1] + self.radius, 1.0))
        self.canvas.ellipse(xy=boundary, fill=self.draw_color)

    def _clip(self, a, a_min, a_max):
        """ clip a into a_min~a_max
        when a out of range
        it will return 0.
        :param a: 
        :param a_min: 
        :param a_max: 
        :return: 
        """
        return a * (a >= a_min and a <= a_max)

    def _gen_next_dir(self, dir):
        # 角度单位是弧度

        next_dir = dir + self._clip(vonmises_line.rvs(self.angle_shape, self.angle_loc, self.angle_scale),
                                    self.angle_diff_min, self.angle_diff_max)

        '''
        next_dir = dir + np.clip(norm.rvs(self.angle_loc_norm, self.angle_scale_norm), a_min=self.angle_diff_min, a_max=self.angle_diff_max)
        '''
        if next_dir > np.pi:
            next_dir = next_dir % np.pi - np.pi
        elif next_dir < - np.pi:
            next_dir = next_dir % - np.pi + np.pi
        else:
            pass

        return next_dir
        # return dir + np.clip(norm.rvs(self.angle_loc_norm, self.angle_scale_norm), a_min=self.angle_diff_min, a_max=self.angle_diff_max)

    def _forward_dynamics(self, action, debug=False):
        # 先根据相对随机游走生成下一步的方向
        self.dir = self._gen_next_dir(self.dir)
        # 获取action中的参数
        _periodcount, _stimulateside = action
        # 新的角度等于(刺激产生的偏转角+原始角度)与随机产生的角度直接求取平均值
        # 用作Debug
        pre_dir = self.dir
        # 第一步的方向为初始化方向，不进行刺激动作

        # 同侧:异侧 = 8：2 偏转，需根据实际统计数值进行修改
        '''
        randint = np.random.randint(1, 10, 1)
        if randint > 8:
            if _stimulateside == 0:
                _stimulateside = 1
            else:
                _stimulateside = 0
        '''
        # 目前仅考虑周期个数对direction的影响，并且假设为线性关系,偏转角度小于等于90度，同侧偏转和异侧偏转比例暂设置为8：2
        # 左侧偏转即现在方向沿逆时针旋转，右侧偏转为顺时针
        #
        # 左侧偏转
        if _stimulateside == 1:
            # the direction is only depended on the periodcount
            # walker.direction = (cur_direction + np.pi / (2. * self.pc_ul) * periodcount + cur_direction) / 2.
            self.dir = self.dir + (np.pi / 2.) * (_periodcount / self.pc_ul)
            # 处理方向角大于 pi的情况
            if self.dir > np.pi:
                self.dir = self.dir % np.pi - np.pi

        # 右侧偏转
        elif _stimulateside == 2:
            # walker.direction = (cur_direction - np.pi / (2. * self.pc_ul) * periodcount + cur_direction) / 2.
            self.dir = (self.dir - np.pi / (2. * self.pc_ul) * _periodcount)
            # 处理方向角小于 - pi的情况
            if self.dir < -np.pi:
                self.dir = self.dir % -np.pi + np.pi

        else:
            # print("stimulate side %d is not in valid range" % _stimulateside)
            pass

        # 用作DEBUG
        pos_dir = self.dir

        if debug:
            print("pre_direction", pre_dir)
            print("action", action)
            # print("processed_action", processed_action)
            print("post_direction", pos_dir)


class BeeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'state']}

    def __init__(
            self,
            screen_width=600,
            screen_height=600,
            periodcount_uplimit=10,
            periodcount_lowlimit=0,
            max_step=1000,
            dutycycle_uplimit=0,
            frequency_uplimit=0,
            dutycycle_lowlimit=0,
            frequency_lowlimit=0,
            path_set_way="auto",
            path_p_random_select=False,
            path_p_idx=None,
            p_dis_range=None,
            path_p_num=None,
            path_fname=None,
            lives=10,
            reflect=False,
            print_info=False,
            debug=False,
    ):
        """
        
        :param screen_width: 
        :param screen_height: 
        :param path_point_num: 
        :param periodcount_uplimit: 
        :param periodcount_lowlimit: 
        :param max_step: 
        :param dutycycle_uplimit: 
        :param frequency_uplimit: 
        :param dutycycle_lowlimit: 
        :param frequency_lowlimit: 
        :param print_info: 
        :param debug: 
        """

        self.viewer = None
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.state_width = self.screen_width / 6
        self.state_height = self.screen_height / 6
        self.screen_dim = (screen_width, screen_height)
        # bee's size = 0.03 * screen_width
        self.player_radius = percent_round_int(screen_width, 0.015)
        self.target_radius = percent_round_int(screen_width, 0.03)
        self.player_size = 2 * self.player_radius
        self.player_color = (0.55, 0.55, 0.55)
        self.target_color = (0.85, 0.85, 0.85)

        self.INIT_POS = (screen_width / 2, screen_height / 2)
        self.INIT_DIR = 0
        self.NOOP = 0
        self._t_lives = lives
        self.lives = lives
        self.BG_COLOR = (255, 255, 255)

        """
        dc -> duty cycle
        fre -> frequency
        pc -> period count
        ss -> stimulate side
        ll -> low limit
        ul -> up limit
        """

        self.dc_ul = dutycycle_uplimit
        self.dc_ll = dutycycle_lowlimit
        self.fre_ul = frequency_uplimit
        self.fre_ll = frequency_lowlimit
        self.pc_ul = periodcount_uplimit
        self.pc_ll = periodcount_lowlimit
        self.ss_ll = 1  # 0 represents NOOP 1 represents left, 2 represents right
        self.ss_ul = 2
        self.para_step = 10

        self._dc_list = [x for x in range(self.dc_ll, self.dc_ul + 1, self.para_step)]
        self._fre_list = [x for x in range(self.fre_ll, self.fre_ul + 1, self.para_step)]
        self._pc_list = [x for x in range(self.pc_ll, self.pc_ul + 1, self.para_step)]
        self._ss_list = [x for x in range(self.ss_ll, self.ss_ul + 1, 1)]

        self._action_map = self._action_map_gen(pc=True, ss=True)

        self._printInfo = print_info
        self._debug = debug
        self._reflect = reflect

        # initialize experiment arena
        self._path_set_way = path_set_way
        self._target_list = []
        self._max_step = max_step
        self._path_p_random_select = path_p_random_select
        if self._path_set_way == "auto":
            if p_dis_range == None:
                raise ValueError("In auto path set way, the p_dis_range is needed but got None")
            if path_p_num == None:
                raise ValueError("In auto path set way, the path_p_num is needed but got None")

            self._path_p_num = path_p_num
            self._p_dis_range = [p_dis_range[0] * screen_width, p_dis_range[1] * screen_width]

        elif self._path_set_way == "load_pre":
            if path_fname == None:
                raise ValueError("In load pre path way, the path_fname is needed but got None")

            self._path_fname = path_fname
            self._path_p_random_select = path_p_random_select
            if not self._path_p_random_select:
                self._path_p_idx = path_p_idx
                if self._path_p_idx == None:
                    raise ValueError("If path_p_random_select is set to False, the path_p_idx is needed but get None")

        self.rewards = {
            "positive": 0.1,
            "negative": -0.1,
            "tick": -0.001,
            "loss": -10.0,
            "win": 10.0,
            "dis_closer": 0.01,
            "dis_away": -0.01,
            "nochange": 0.0
        }

    def _action_map_gen(self, dc=False, fre=False, pc=False, ss=False):
        action_map = {}
        param_list = []
        if dc:
            param_list.append(self._dc_list)
        if fre:
            param_list.append(self._fre_list)
        if pc:
            param_list.append(self._pc_list)
        if ss:
            param_list.append(self._ss_list)

        from itertools import product
        param_pair = [list(x) for x in list(product(*param_list))]
        for i in range(len(param_pair)):
            action_map.update({i: param_pair[i]})

        return action_map

    '''================================target list generation functions====================================='''

    def _load_pre_set_path(self):
        import joblib
        path_file = joblib.load(self._path_fname)
        path_type = path_file["type"]
        path_point = path_file["path_pos"]
        self.INIT_POS = path_file["init_pos"]
        self.INIT_DIR = path_file["init_dir"]
        if path_type == "point":
            if self._path_p_random_select:
                idx = np.random.randint(0, len(path_point))
            else:
                if self._path_p_idx >= len(path_point):
                    raise ValueError("Path_p_idx:%d is out of range(0,%d)" % (self._path_p_idx, len(path_point)))
                idx = self._path_p_idx
            pos = path_point[idx]["pos"]
            radius = self.target_radius
            color = self.target_color
            self._target_list.append(Target(pos_init=pos, idx=None, color=color, radius=radius,
                                            SCREEN_WIDTH=self.screen_width, SCREEN_HEIGHT=self.screen_height))
        else:
            for idx, p in enumerate(path_point):
                pos = p["pos"]
                radius = self.target_radius
                color = self.target_color
                # radius = p["radius"][idx]
                # color = p["color"][idx]
                self._target_list.append(Target(pos_init=pos, idx=idx, color=color, radius=radius,
                                                SCREEN_WIDTH=self.screen_width, SCREEN_HEIGHT=self.screen_height))

        self._path_p_num = len(self._target_list)
        #print("%d point(s) add to target list" % (len(self._target_list)))

    def _auto_set_path_point(self):
        self._target_list.append(
            Target(pos_init=self.INIT_POS, idx=0, color=self.target_color, radius=self.target_radius,
                   SCREEN_WIDTH=self.screen_width, SCREEN_HEIGHT=self.screen_height))
        while len(self._target_list) < self._path_p_num:
            # add init pos target to target list
            pre_coor = np.array([self._target_list[-1].pos.x, self._target_list[-1].pos.y])
            rand_dist = np.random.uniform(self._p_dis_range[0], self._p_dis_range[1])
            rand_dire = np.random.uniform(-1, 1) * np.pi
            cur_coor = pre_coor + np.array([rand_dist * np.cos(rand_dire), rand_dist * np.sin(rand_dire)])
            # print(pre_coor, cur_coor)
            # check if hit boundaries
            if self._check_hit_boundary(cur_coor):
                # print("hit boundary\n")
                continue
            # check if overlap
            if self._check_overlap(cur_coor):
                # print("overlap\n")
                continue
            pos = (cur_coor[0], cur_coor[1])
            color = self.target_color
            radius = self.target_radius
            self._target_list.append(Target(pos_init=pos, idx=len(self._target_list), color=color, radius=radius,
                                            SCREEN_WIDTH=self.screen_width, SCREEN_HEIGHT=self.screen_height))

    def _check_hit_boundary(self, cur_coor):
        if (cur_coor[0] > self.screen_width - self.target_radius) or (cur_coor[0] < self.target_radius) or cur_coor[
            1] > self.screen_height - self.target_radius or cur_coor[1] < self.target_radius:
            return True
        else:
            return False

    def _check_overlap(self, cur_coor):
        """
        :param cur_coor: the current point's coordinate of the path
        :return: True if the overlap otherwise False
        """
        for target in self._target_list:
            if vec2d(cur_coor).dist(target.pos) <= 2 * target.radius:
                return True
        else:
            return False

    def _check_length_in_range(self, pre_coor, cur_coor):
        """
        :param pre_coor: the coordinate of the previous point in the path
        :type np.array
        :param cur_coor: the coordinate of current point in the path
        :type np.array
        :return: True if the length of vector(cur_coor - pre_coor) in the of length range otherwise return false
        """
        vec_norm = vec2d(cur_coor).dist(pre_coor)
        if vec_norm < self._p_dis_range[0] or vec_norm > self._p_dis_range[1]:
            return False
        else:
            return True

    def _collide_circle(self, left, right):
        """detect collision between two sprites using circles
        Tests for collision between two sprites by testing whether two circles
        centered on the sprites overlap. If the sprites have a "radius" attribute,
        then that radius is used to create the circle; otherwise, a circle is
        created that is big enough to completely enclose the sprite's rect as
        given by the "rect" attribute. This function is intended to be passed as
        a collided callback function to the *collide functions. Sprites must have a
        "rect" and an optional "radius" attribute.
        """

        xdistance = left.center[0] - right.center[0]
        ydistance = left.center[1] - right.center[1]
        distancesquared = xdistance ** 2 + ydistance ** 2

        if hasattr(left, 'radius'):
            leftradius = left.radius
        else:
            leftrect = left.rect
            # approximating the radius of a square by using half of the diagonal,
            # might give false positives (especially if its a long small rect)
            leftradius = 0.5 * ((leftrect[0] ** 2 + leftrect[0] ** 2) ** 0.5)
            # store the radius on the sprite for next time
            setattr(left, 'radius', leftradius)

        if hasattr(right, 'radius'):
            rightradius = right.radius
        else:
            rightrect = right.rect
            # approximating the radius of a square by using half of the diagonal
            # might give false positives (especially if its a long small rect)
            rightradius = 0.5 * ((rightrect[0] ** 2 + rightrect[1] ** 2) ** 0.5)
            # store the radius on the sprite for next time
            setattr(right, 'radius', rightradius)
        return distancesquared <= (leftradius + rightradius) ** 2

    '''================================custom env functions====================================='''

    def _compute_reward(self):
        # print(self.passed_path_point)
        tick_reward = self.rewards["tick"]
        hit = self._collide_circle(self.player, self._target_list[self._passed_path_point])
        if hit:  # it hit
            self.score += self.rewards["positive"] * self._passed_path_point
            self._passed_path_point += 1

        if self.finish:
            done_reward = self.rewards["win"]
            self.score += done_reward
            return done_reward
        elif self.game_over:
            loss_reward = self.rewards["loss"]
            self.score += loss_reward
            return loss_reward
        else:
            next_dest_target = self._target_list[self._passed_path_point]
            next_dest_coordinate = next_dest_target.pos
            pre_coordinate = self.player.pre_pos
            cur_coordinate = self.player.pos

            pre_distance = next_dest_coordinate.dist(pre_coordinate)
            cur_distance = next_dest_coordinate.dist(cur_coordinate)

            if pre_distance > cur_distance:
                distance_reward = self.rewards["dis_closer"]
            elif pre_distance == cur_distance:
                distance_reward = self.rewards["nochange"]
            else:
                distance_reward = self.rewards["dis_away"]

            reward = distance_reward + tick_reward
            self.score += reward

            return reward

            # angle_reward = np.cos(compute_angle(next_dest_point - pre_coordinate, cur_coordinate - pre_coordinate))

    def _check_hit(self):
        # boundary check
        x_check = (
                      self.player.pos.x < 0) or \
                  (
                      self.player.pos.x +
                      self.player_size
                      > self.screen_width)
        y_check = (
                      self.player.pos.y < 0) or \
                  (
                      self.player.pos.y +
                      self.player_size
                      > self.screen_height)

        return x_check or y_check

    def _boundary_reflect(self):
        if self.player.pos.x > self.screen_width - self.player_size:
            if self._debug:
                print("right boundy, pos: x=%f,y=%f" % (self.player.pos.x, self.player.pos.y))
            self.player.pos.x = 2 * (self.screen_width - self.player_size) - self.player.pos.x
            self.player.dir = np.pi - self.player.dir

        elif self.player.pos.x < 0:
            if self._debug:
                print("left boundy, pos: x=%f,y=%f, size=%f" % (self.player.pos.x, self.player.pos.y, self.player_size))
            self.player.pos.x = - self.player.pos.x
            self.player.dir = np.pi - self.player.dir

        if self.player.pos.y > self.screen_height - self.player_size:
            if self._debug:
                print("botton boundy, pos: x=%f,y=%f" % (self.player.pos.x, self.player.pos.y))
            self.player.pos.y = 2 * (self.screen_height - self.player_size) - self.player.pos.y
            self.player.dir = - self.player.dir

        elif self.player.pos.y < 0:
            if self._debug:
                print("up boundy, pos: x=%f,y=%f, size=%f" % (self.player.pos.x, self.player.pos.y, self.player_size))
            self.player.pos.y = - self.player.pos.y
            self.player.dir = - self.player.dir

        if self._debug:
            print("player's pos is x:%f, y:%f, direction:%f degree" % (
                self.player.pos.x, self.player.pos.y, self.player.dir / np.pi * 180))

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
        # pygame init
        # pygame.init()
        # self.screen = pygame.display.set_mode(self.getScreenDims(), 0, 32)

        # init target list
        self._target_list.clear()
        if self._path_set_way == "auto":
            self._auto_set_path_point()
        elif self._path_set_way == "load_pre":
            self._load_pre_set_path()
        else:
            raise ValueError("The path set way is invalid")

        # init bee player
        self.player = BeePlayer(
            self.INIT_DIR,
            self.INIT_POS,
            self.player_radius,
            self.player_color,
            self.screen_width,
            self.screen_height,
            (self.dc_ul, self.dc_ll, self.fre_ul, self.fre_ul, self.fre_ll, self.pc_ul, self.pc_ll, self.ss_ul,
             self.ss_ll),
        )

        self.score = 0
        self.ticks = 0
        self.lives = self._t_lives

        self._passed_path_point = 0

        if len(self._target_list) == 0:
            raise ValueError("the route path is empty")
            # self.screen.fill(self.BG_COLOR)
            # pygame.display.flip()

    def getScreenRGB(self):
        """
        Returns the current game screen in RGB format.

        Returns
        --------
        numpy uint8 array
            Returns a numpy array with the shape (width, height, 3).

        """
        return self._render(mode="rgb_array")

    def getScreenDims(self):
        """
        Gets the screen dimensions of the game in tuple form.

        Returns
        -------
        tuple of int
            Returns tuple as follows (width, height).

        """
        return self.screen_dim

    def getGameState(self):
        """
        Returns
        -------

        dict
            * bee x position.
            * bee y position.
            * bee head angle
            * target x position.
            * target y position.
            See code for structure.

        """
        if self._passed_path_point == len(self._target_list):
            idx = self._passed_path_point - 1
        else:
            idx = self._passed_path_point
        state = {
            "bee_x": self.player.pos.x,
            "bee_y": self.player.pos.y,
            "bee_head_angle": self.player.dir,
            "target_x": self._target_list[idx].pos.x,
            "target_y": self._target_list[idx].pos.y
        }

        return state

    def getScore(self):
        return self.score

    '''================================override gym function ====================================='''

    def _step(self, action):
        #whether the tick num > max_step
        if self.ticks > self._max_step:
            self.lives = -1
        else:
            self.ticks += 1
            if isinstance(action, np.ndarray):
                action = action.item()
            _real_action = self.action_space.mapping[action]
            self.player.update(_real_action)
            # 碰壁就算失败
            if self._check_hit():
                if self._reflect:
                    self._boundary_reflect()
                    self.lives -= 1
                else:
                    self.lives = -1

        reward = self._compute_reward()
        self.score += reward
        observation = self.render("state")
        info = self.getGameState()

        return observation, reward, self.terminal, info

    def _render(self, mode="human", close=False):
        if mode == "human" or mode == "rgb_array":
            from .utils import rendering
            if close:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                return

            if self.viewer is None:
                if mode == "human":
                    self.viewer = rendering.Viewer(self.screen_width, self.screen_height, visible=True)
                else:
                    self.viewer = rendering.Viewer(self.screen_width, self.screen_height, visible=False)

            # draw the targets
            for i in range(self._passed_path_point, len(self._target_list)):
                self._target_list[i].render(self.viewer)
            # draw the player
            self.player.render(self.viewer)
            return self.viewer.render(return_rgb_array=mode == "rgb_array")
        elif mode == "state":
            from PIL import Image
            self.arr = Image.new("RGB", (self.screen_width, self.screen_height), self.BG_COLOR)
            for i in range(self._passed_path_point, len(self._target_list)):
                self._target_list[i].draw(self.arr)
            self.player.draw(self.arr)
            return np.array(self.arr)
        else:
            raise NotImplemented

    def _reset(self):
        """
        Performs a reset of the games to a clean initial state.
        """
        self.init()
        return self.render("state")

    '''================================some property functions ====================================='''

    @property
    def game_over(self):
        return self.lives == -1

    @property
    def finish(self):
        return len(self._target_list) == self._passed_path_point

    @property
    def terminal(self):
        return self.finish or self.game_over

    @property
    def action_space(self):
        return DiscreteToMultiDiscrete(spaces.MultiDiscrete([[self.pc_ll, self.pc_ul], [self.ss_ll, self.ss_ul]]),
                                       self._action_map)

    @property
    def observation_space(self):
        # min_x, max_x, min_y, max_y = 0, self.screen_width, 0, self.screen_height
        # return spaces.Box(np.array([min_x, min_y]), np.array([max_x, max_y]))
        return spaces.Box(low=0.0, high=255.0, shape=(self.screen_height, self.screen_width, 3))
