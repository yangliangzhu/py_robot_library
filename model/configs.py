import numpy as np
from numpy import pi
from .dh_param import *
from copy import deepcopy
# todo 移动到model文件夹下

# * example: franka panda, mdh
dh_franka_mdh = Dh([
    [333, 0, 0.],
    [0., -pi/2, 0.],
    [316, pi/2, 0.],
    [0., pi/2, 82.5],
    [384, -pi/2, -82.5],
    [0., pi/2, 0.],
    [0., pi/2, 88]
])
dh_franka_mdh.millimeter_to_meter()
franka_upper = np.radians([175, 135, 140, 175, 175, 175, 175])
franka_lower = np.radians([-175, -135, -170, -175, -175, -175, -175])
franka_config = {
    "type": "mdh_param",
    "param": dh_franka_mdh,
    "upper": franka_upper,
    "lower": franka_lower,
}


# * example: rokae xMateER3, sdh
dh_rokae_er3_sdh = Dh([
    [341.5, -pi/2, 0.],
    [0.0, pi/2, 0.],
    [394.0, -pi/2, 0.],
    [0.0, pi/2, 0.],
    [366.0, -pi/2, 0.],
    [0.0, pi/2, 0.],
    [250.3, 0., 0.],
], type='sdh')
dh_rokae_er3_sdh.millimeter_to_meter()
rokae_er3_upper = np.radians([165, 115, 165, 115, 165, 115, 355])
rokae_er3_lower = np.radians([-165, -115, -165, -115, -165, -115, -355])
rokae_er3_config_sdh = {
    "type": "sdh_param",
    "param": dh_rokae_er3_sdh,
    "upper": rokae_er3_upper,
    "lower": rokae_er3_lower,
}

# * example: rokae xMateER3, mdh
dh_rokae_er3_mdh = Dh([
    [341.5, 0, 0.],
    [0.0, -pi/2, 0.],
    [394.0, pi/2, 0.],
    [0.0, -pi/2, 0.],
    [366.0, pi/2, 0.],
    [0.0, -pi/2, 0.],
    [250.3, pi/2, 0.],
])
dh_rokae_er3_mdh.millimeter_to_meter()
rokae_er3_config = {
    "type": "mdh_param",
    "param": dh_rokae_er3_mdh,
    "upper": rokae_er3_upper,
    "lower": rokae_er3_lower,
}

rokae_er3_upper_expand = np.radians([170, 120, 170, 130, 170, 120, 360])
rokae_er3_lower_expand = np.radians([-170, -120, -170, -120, -170, -120, -360])

rokae_er3_config_expand = {
    "type": "mdh_param",
    "param": dh_rokae_er3_mdh,
    "upper": rokae_er3_upper_expand,
    "lower": rokae_er3_lower_expand,
}

# * example: xMateER3 Plus version
dh_rokae_er3_mdh_plus = Dh([
    [341.5, 0, 0.],
    [0.0, -pi/2, 0.],
    [444.0, pi/2, 0.],
    [0.0, -pi/2, 0.],
    [416.0, pi/2, 0.],
    [0.0, -pi/2, 0.],
    [250.3, pi/2, 0.],
])
dh_rokae_er3_mdh_plus.millimeter_to_meter()
rokae_er3_config_plus = {
    "type": "mdh_param",
    "param": dh_rokae_er3_mdh_plus,
    "upper": rokae_er3_upper_expand,
    "lower": rokae_er3_lower_expand,
}

# * example: NervER3 version
dh_nerv_er3_mdh = Dh([
    [341.5, 0, 0.],
    [0.0, -pi/2, 0.],
    [444.0, pi/2, 0.],
    [0.0, -pi/2, 0.],
    [416.0, pi/2, 0.],
    [97.0, -pi/2, 0.],  # 0.037 + 0.06
    [114.8, pi/2, 0.],
])
dh_nerv_er3_mdh.millimeter_to_meter()
nerv_er3_config = {
    "type": "mdh_param",
    "param": dh_nerv_er3_mdh,
    "upper": rokae_er3_upper_expand * 3,
    "lower": rokae_er3_lower_expand * 3,
}

# * rokae xMateSR5
trans01 = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=float)

trans12 = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0.328], [0, 0, 0, 1]], dtype=float)

trans23 = np.array(
    [[1, 0, 0, 0.05], [0, -1, 0, -0.4], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=float)

trans34 = np.array(
    [[1, 0, 0, -0.05], [0, 0, 1, 0.4], [0, -1, 0, 0.0], [0, 0, 0, 1]],
    dtype=float)

trans45 = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0.136], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=float)

trans56 = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0.1035], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=float)

sr5_mat_list = [trans01, trans12, trans23, trans34, trans45, trans56]
rokae_sr5_upper = np.radians([360, 150, 140, 360, 360, 360])
rokae_sr5_lower = np.radians([-360, -160, -170, -360, -360, -360])
rokae_sr5_config = {
    "type": "mat",
    "param": sr5_mat_list,
    "upper": rokae_sr5_upper,
    "lower": rokae_sr5_lower,
}

# * example: NervER3 version
dh_nerv_er3_mdh = Dh([
    [341.5, 0, 0.],
    [0.0, -pi/2, 0.],
    [444.0, pi/2, 0.],
    [0.0, -pi/2, 0.],
    [416.0, pi/2, 0.],
    [97.0, -pi/2, 0.],  # 0.037 + 0.06
    [114.8, pi/2, 0.],
])
dh_nerv_er3_mdh.millimeter_to_meter()
nerv_er3_config = {
    "type": "mdh_param",
    "param": dh_nerv_er3_mdh,
    "upper": rokae_er3_upper_expand * 3,
    "lower": rokae_er3_lower_expand * 3,
}


if __name__ == '__main__':
    print(sr5_mat_list)
