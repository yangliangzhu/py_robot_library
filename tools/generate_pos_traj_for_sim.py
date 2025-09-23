import numpy as np
from enum import Enum

from tools.geometry import *


class ToolType(Enum):

    AilanRf = "ailan_radio_freq"
    NervRf = "nerv_radio_freq"
    AilanSw = "ailan_shockwave"
    NervSw = "nerv_shockwave"  # also test_sw


def select_connector(dof):
    res = np.eye(4)
    if dof == 7:
        res[2, 3] = 0.0736
    elif dof == 6:
        res[2, 3] = 0.0736
        res[0, 3] = 0.039
    return res


def select_tool(dof, tool_name):
    connector = select_connector(dof)
    res = np.eye(4)
    if tool_name == ToolType.AilanRf:
        res[2, 3] = 0.0824
    elif tool_name == ToolType.NervRf:
        res[2, 3] = 0.07224
    elif tool_name == ToolType.AilanSw:
        res[0, 3] = 0.085
        res[2, 3] = 0.123
    elif tool_name == ToolType.NervSw:
        res[0, 3] = 0.02
        res[1, 3] = 0.08
        res[2, 3] = 0.09

    return connector @ res


def generate_npz(data, tool_name):
    p_cmd = data['p_cmd']  # * 已经是末端工具位置
    q_cmd = data['q_cmd']
    dof = 6 if np.max(np.abs(q_cmd[:, -1])) < 1e-5 else 7
    tool_trans = select_tool(dof, tool_name)

    data_num = p_cmd.shape[0]
    cmd_pose = []
    for i in range(data_num):
        pose = nervCartToAffine(p_cmd[i])
        cmd_pose.append(pose.flatten())

    extra_data_num = 400
    last_data = nervCartToAffine(p_cmd[-1])
    print(last_data)
    y_vel = 0.05
    dt = 0.008
    for i in range(extra_data_num):
        delta_y = y_vel * dt * (i+1)
        data_extra = np.array(last_data)
        data_extra[1, 3] += delta_y
        cmd_pose.append(data_extra.flatten())

    np.savez('npz/sim_pose.npz', p=cmd_pose, q=q_cmd, tool=tool_trans)


if __name__ == '__main__':
    data = np.load('npz/cmd.npz')
    tool_type = ToolType.NervRf
    generate_npz(data, tool_type)
