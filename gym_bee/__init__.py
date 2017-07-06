#!/usr/bin/env python
# coding=utf-8
from gym.envs.registration import register

register(
        id='bumblebee-v0',
        entry_point='gym_bumblebee.envs:BeeEnv',
        )
