''' 
for debug new ik
'''
from tools.geometry import *
import numpy as np
import matplotlib.pyplot as plt
from model.model_factory import ModelFactory

# plt.style.use('seaborn-v0_8')


def view_manip(data):
    # * notice it's in degrees
    q_cmd = np.deg2rad(data['q_cmd'])
    t_qc = data['t_qc']

    dof = 6 if np.max(np.abs(q_cmd[:, -1])) < 1e-5 else 7
    model_factory = ModelFactory()
    model_sr5 = model_factory.create('sr5')
    model_er3 = model_factory.create('er3')
    model = model_er3 if dof == 7 else model_sr5

    data_num = q_cmd.shape[0]
    manips = []
    for i in range(data_num):
        manips.append(model.manip(q_cmd[i, :dof]))

    plt.plot(t_qc, np.log10(manips), alpha=0.7, label=f'log10(manip)')
    plt.legend()
    plt.show()


def view_norm_degrees(data):
    '''
    分析J4引起的操作度问题
    '''
    z_axis = np.array([0, 0, -1])
    p_cmd = data['p_cmd']
    p_act = data['p_act']
    t_pc = data['t_pc']
    t_pa = data['t_pa']

    data_num = p_cmd.shape[0]
    angles_act = []
    angles_cmd = []
    for i in range(data_num):
        z_act_i = nervRpy(p_act[i])[:3, 2]
        z_cmd_i = nervRpy(p_cmd[i])[:3, 2]
        sin_angle_vec_act = np.cross(z_act_i, z_axis)
        sin_angle_vec_cmd = np.cross(z_cmd_i, z_axis)
        sin_angle_act = np.linalg.norm(sin_angle_vec_act)
        sin_angle_cmd = np.linalg.norm(sin_angle_vec_cmd)
        if sin_angle_vec_act[0] > 0:
            sin_angle_act = -sin_angle_act
        if sin_angle_vec_cmd[0] > 0:
            sin_angle_cmd = -sin_angle_cmd
        angles_act.append(np.arcsin(sin_angle_act))
        angles_cmd.append(np.arcsin(sin_angle_cmd))

    plt.plot(t_pa, np.degrees(angles_act),
             alpha=0.7, label=f'Norm Vector Angle Act')
    plt.plot(t_pc, np.degrees(angles_cmd),
             alpha=0.7, label=f'Norm Vector Angle Cmd')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = np.load('npz/cmd.npz')

    # * check manip
    view_manip(data)

    # * check J4 angles
    view_norm_degrees(data)
