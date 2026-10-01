"""Rigid-body motion helpers.

Screw-theoretic utilities in the style of Murray, Li and Sastry,
*A Mathematical Introduction to Robotic Manipulation* (1994): SO(3)/SE(3)
exponentials and logarithms, adjoint and ``ad`` operators, and forward and
inverse dynamics for an open chain.

These are low-level, convention-heavy helpers. The library's own robot models do
not depend on them; they are provided as building blocks for custom analyses.
"""

import numpy as np

# ---------------------------------------------------------------------------
# general functions
# ---------------------------------------------------------------------------


def near_zero(z):
    # 检查是否接近0
    return abs(z) < 1e-6


def normalize(v):
    # 模长归一化
    return v / np.linalg.norm(v)


'''
rigid body motions
'''


def rot_inv(R):
    # 旋转矩阵求逆
    return np.array(R).T


def vec_to_so3(omg):
    # 向量转化为so3
    return np.array([[0, -omg[2], omg[1]], [omg[2], 0, -omg[0]],
                     [-omg[1], omg[0], 0]])


def so3_to_vec(so3mat):
    # so3转化为向量
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])


def axis_ang3(expc3):
    # 将向量化为转轴-转角格式
    return (normalize(expc3), np.linalg.norm(expc3))


def matrix_exp3(so3mat):
    # so3映射到SO3
    omgtheta = so3_to_vec(so3mat)
    if near_zero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = axis_ang3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
            + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)


def matrix_log3(R):
    # SO3映射到so3
    acos_input = (np.trace(R) - 1) / 2.0
    if acos_input >= 1:
        return np.zeros((3, 3))
    elif acos_input <= -1:
        if not near_zero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not near_zero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return vec_to_so3(np.pi * omg)
    else:
        theta = np.arccos(acos_input)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)


def Rp_to_trans(R, p):
    # 旋转和平移合成刚体运动
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]


def trans_to_Rp(T):
    # 刚体运动拆分为旋转和平移
    T = np.array(T)
    return T[:3, :3], T[:3, 3]


def trans_inv(T):
    # 刚体运动求逆
    R, p = trans_to_Rp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]


def vec_to_se3(v):
    # 向量转化为se3
    return np.r_[np.c_[vec_to_so3(v[:3]), v[3:]], np.zeros((1, 4))]


def se3_to_vec(se3mat):
    # se3转化为向量
    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                 [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]


def adjoint(T):
    # 伴随矩阵
    R, p = trans_to_Rp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))], np.c_[np.dot(vec_to_so3(p), R),
                                                   R]]


def screw_to_axis(q, s, h):
    # 螺旋描述转为运动旋量正则化
    return np.r_[s, np.cross(q, s) + np.dot(h, s)]


def axis_ang6(expc6):
    # 将向量化为螺旋轴-螺旋量格式
    theta = np.linalg.norm([expc6[0], expc6[1], expc6[2]])
    if near_zero(theta):
        theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]])
    return (np.array(expc6 / theta), theta)


def matrix_exp6(se3mat):
    # se3到SE3
    se3mat = np.array(se3mat)
    omgtheta = so3_to_vec(se3mat[0:3, 0:3])
    if near_zero(np.linalg.norm(omgtheta)):
        return Rp_to_trans(np.eye(3), se3mat[:3, 3])
    else:
        theta = axis_ang3(omgtheta)[1]
        omgmat = se3mat[0:3, 0:3] / theta
        rotation = matrix_exp3(se3mat[0:3, 0:3])
        coeff = (np.eye(3) * theta + (1 - np.cos(theta)) * omgmat +
                 (theta - np.sin(theta)) * omgmat @ omgmat) / theta
        translation = coeff @ se3mat[:3, 3]
        return Rp_to_trans(rotation, translation)


def matrix_log6(T):
    # SE3到se3
    R, p = trans_to_Rp(T)
    omgmat = matrix_log3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)), p], [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        coeff = np.eye(3) - omgmat / 2 + (
            1 - theta / 2 / np.tan(theta / 2)) * omgmat @ omgmat / theta**2
        trans = coeff @ p

        return np.r_[np.c_[omgmat, trans], [[0, 0, 0, 0]]]


def ad(v):
    # SE3的李括号
    w_mat = vec_to_so3(v[:3])
    res = np.r_[np.c_[w_mat, np.zeros((3, 3))],
                np.c_[vec_to_so3(v[3:]), w_mat]]
    return res


def project_to_SO3(mat):
    # 将矩阵投影到最近的SO3矩阵
    U, s, Vh = np.linalg.svd(mat)
    R = np.dot(U, Vh)
    if np.linalg.det(R) < 0:
        R[:, s[2, 2]] = -R[:, s[2, 2]]
    return R


def project_to_SE3(mat):
    # 将矩阵投影到最近的SE3矩阵
    mat = np.array(mat)
    return Rp_to_trans(project_to_SO3(mat[:3, :3]), mat[:3, 3])


def distance_to_SO3(mat):
    # 返回衡量矩阵到SO3流形的Frobenius范数
    mat = np.array(mat)
    if np.linalg.det(mat) > 0:
        return np.linalg.norm(mat.T @ mat - np.eye(3))
    else:
        return 1e+9


def distance_to_SE3(mat):
    # 返回衡量矩阵到SE3流形的Frobenius范数
    mat = np.array(mat)
    matR = mat[:3, :3]
    if np.linalg.det(matR) > 0:
        mat_diff = np.array(mat)
        mat_diff[:3, :3] = matR.T @ matR
        mat_diff[:3, 3] = np.zeros(3)
        return np.linalg.norm(mat_diff - np.eye(4))
    else:
        return 1e+9


def test_if_SO3(mat):
    # 检查是否接近SO3
    return abs(distance_to_SO3(mat)) < 1e-3


def test_if_SE3(mat):
    # 检查是否接近SO3
    return abs(distance_to_SE3(mat)) < 1e-3


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------


def fk_in_space(mat, s_list, q_list):
    # s_list取列
    res = np.array(mat)
    for i in range(len(q_list) - 1, -1, -1):
        res = matrix_exp6(vec_to_se3(s_list[:, i] * q_list[i])) @ res
    return res


def fk_in_body(mat, b_list, q_list):
    # s_list取列
    res = np.array(mat)
    for i in range(len(q_list)):
        res = res @ matrix_exp6(vec_to_se3(b_list[:, i] * q_list[i]))
    return res


def b2s_list(mat, b_list):
    # 旋量转换
    s_list = np.zeros_like(b_list)
    ad_mat = adjoint(mat)

    for i in range(b_list.shape[1]):
        s_list[:, i] = ad_mat @ b_list[:, i]
    return s_list


def s2b_list(mat, s_list):
    # 旋量转换
    b_list = np.zeros_like(s_list)
    ad_mat = adjoint(trans_inv(mat))

    for i in range(s_list.shape[1]):
        b_list[:, i] = ad_mat @ s_list[:, i]
    return b_list


def jacobian_space(s_list, q_list):
    # 空间雅克比
    jac = np.array(s_list)
    T = np.eye(4)
    for i in range(1, len(q_list)):
        T = T @ matrix_exp6(vec_to_se3(s_list[:, i-1] * q_list[i-1]))
        jac[:, i] = adjoint(T) @ s_list[:, i]
    return jac


def jacobian_body(b_list, q_list):
    # 物体雅克比
    jac = np.array(b_list)
    T = np.eye(4)
    for i in range(len(q_list)-2, -1, -1):
        T = T @ matrix_exp6(vec_to_se3(- b_list[:, i+1] * q_list[i+1]))
        jac[:, i] = adjoint(T) @ b_list[:, i]
    return jac


def ik_in_body(b_list, M, T, q_list0, eomg=0.01, ev=0.001):
    # 基于物体坐标系求反解
    q_list = np.array(q_list0)
    i = 0
    max_iter = 20
    T_now = fk_in_body(M, b_list, q_list)
    vb = se3_to_vec(matrix_log6(trans_inv(T_now) @ T))
    err = np.linalg.norm(vb[:3]) > eomg \
        or np.linalg.norm(vb[3:]) > ev
    while err and i < max_iter:
        q_list += np.linalg.pinv(jacobian_body(b_list,
                                               q_list)) @ vb
        i += 1
        T_now = fk_in_body(M, b_list, q_list)
        vb = se3_to_vec(matrix_log6(trans_inv(T_now) @ T))
        err = np.linalg.norm(vb[:3]) > eomg \
            or np.linalg.norm(vb[3:]) > ev
    return (q_list, not err)


def ik_in_space(s_list, M, T, q_list0, eomg=0.01, ev=0.001):
    # 基于空间坐标系求反解
    q_list = np.array(q_list0)
    i = 0
    max_iter = 20
    T_now = fk_in_space(M, s_list, q_list)
    vb = se3_to_vec(matrix_log6(trans_inv(T_now) @ T))
    vs = adjoint(T_now) @ vb
    err = np.linalg.norm(vs[:3]) > eomg \
        or np.linalg.norm(vs[3:]) > ev
    while err and i < max_iter:
        q_list += np.linalg.pinv(jacobian_space(s_list,
                                                q_list)) @ vs
        i += 1
        T_now = fk_in_space(M, s_list, q_list)
        vb = se3_to_vec(matrix_log6(trans_inv(T_now) @ T))
        vs = adjoint(T_now) @ vb
        err = np.linalg.norm(vs[:3]) > eomg \
            or np.linalg.norm(vs[3:]) > ev
    return (q_list, not err)


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------


def inverse_dynamics_com(q, dq, ddq, g, f_tip, M_list, G_list, s_list):
    """Inverse dynamics with the body frames at the link centres of mass.

    Args:
        q, dq, ddq: Joint positions, velocities and accelerations.
        g: Gravitational acceleration vector.
        f_tip: Wrench applied at the tool tip.
        M_list: Screw axes expressed at the centres of mass, one per joint.
        G_list: Spatial inertia matrices, one per link.
        s_list: Screw axes expressed at the link centres of mass.

    Returns:
        The joint torques, length ``len(q)``.
    """
    n = len(q)
    Mi = np.eye(4)
    Ai = np.zeros((6, n))
    AdTi = [[None]] * (n+1)
    Vi = np.zeros((6, n+1))
    Vdi = np.zeros_like(Vi)
    taus = np.zeros(n)

    Vdi[:, 0] = np.r_[[0, 0, 0], -g]
    AdTi[n] = adjoint(trans_inv(M_list[n]))
    Fi = np.array(f_tip)

    for i in range(n):
        Mi = Mi @ M_list[i]
        Ai[:, i] = adjoint(trans_inv(Mi)) @ s_list[:, i]
        AdTi[i] = adjoint(matrix_exp6(vec_to_se3(-Ai[:, i] * q[i]))
                          @ trans_inv(M_list[i]))
        Vi[:, i+1] = AdTi[i] @ Vi[:, i] + dq[i] * Ai[:, i]
        Vdi[:, i+1] = AdTi[i] @ Vdi[:, i] + ad(Vi[:, i+1]) @ Ai[:, i] * dq[i] \
            + ddq[i] * Ai[:, i]
        # Vdi[:, i+1] = AdTi[i] @ Vdi[:, i] - ad(Ai[:, i] * dq[i]) @ AdTi[i] @ Vi[:, i] \
        #     + ddq[i] * Ai[:, i] #- this is equivalent

    for i in range(n-1, -1, -1):
        Fi = AdTi[i+1].T @ Fi + G_list[i] @ Vdi[:, i+1] \
            - ad(Vi[:, i+1]).T @ (G_list[i] @ Vi[:, i+1])
        taus[i] = np.inner(Fi, Ai[:, i])

    return taus


def inverse_dynamics(q, dq, ddq, g, f_tip, M_list, G_list):
    """Inverse dynamics with the body frames at the link origins.

    Same formulation as :func:`inverse_dynamics_com`, but every screw axis is
    the constant joint axis expressed at the link origin.

    Args:
        q, dq, ddq: Joint positions, velocities and accelerations.
        g: Gravitational acceleration vector.
        f_tip: Wrench applied at the tool tip.
        M_list: Link transforms.
        G_list: Spatial inertia matrices, one per link.

    Returns:
        The joint torques, length ``len(q)``.
    """
    n = len(q)
    Mi = np.eye(4)
    Ai = np.array([0, 0, 1, 0, 0, 0])
    AdTi = [[None]] * (n+1)
    Vi = np.zeros((6, n+1))
    Vdi = np.zeros_like(Vi)
    taus = np.zeros(n)

    Vdi[:, 0] = np.r_[[0, 0, 0], -g]
    AdTi[n] = adjoint(trans_inv(M_list[n]))
    Fi = np.array(f_tip)

    for i in range(n):
        Mi = Mi @ M_list[i]
        AdTi[i] = adjoint(matrix_exp6(vec_to_se3(-Ai * q[i]))
                          @ trans_inv(M_list[i]))
        Vi[:, i+1] = AdTi[i] @ Vi[:, i] + dq[i] * Ai
        Vdi[:, i+1] = AdTi[i] @ Vdi[:, i] + ad(Vi[:, i+1]) @ Ai * dq[i] \
            + ddq[i] * Ai
    print(Vdi[:, i])

    for i in range(n-1, -1, -1):
        Fi = AdTi[i+1].T @ Fi + G_list[i] @ Vdi[:, i+1] \
            - ad(Vi[:, i+1]).T @ (G_list[i] @ Vi[:, i+1])
        taus[i] = np.inner(Fi, Ai)

    return taus


def mass_matrix(q, M_list, G_list, s_list):
    # 惯性矩阵
    n = len(q)
    mass = np.zeros((n, n))
    for i in range(n):
        ddq = np.zeros(n)
        ddq[i] = 1.0
        mass[:, i] = inverse_dynamics(q, np.zeros(n), ddq, np.zeros(3),
                                      np.zeros(6), M_list, G_list, s_list)
    return mass


def coriolis(q, dq, Ms, Gs, ss):
    # 科氏力和离心力
    res = inverse_dynamics(q, dq, np.zeros_like(q), np.zeros(3),
                           np.zeros(6), Ms, Gs, ss)
    return res


def gravity(q, g, Ms, Gs, ss):
    # 重力
    res = inverse_dynamics(q, np.zeros_like(q), np.zeros_like(q), g,
                           np.zeros(6), Ms, Gs, ss)
    return res


def end_effector_force(q, f_tip, Ms, Gs, ss):
    # 外力矩
    res = inverse_dynamics(q, np.zeros_like(q), np.zeros_like(q),
                           np.zeros(3), f_tip, Ms, Gs, ss)
    return res


def forward_dynamics(q, dq, tau, g, f_tip, Ms, Gs, ss):
    # 正向动力学
    acc_tor = tau - coriolis(q, dq, Ms, Gs, ss) - gravity(q, g, Ms, Gs, ss) \
        - end_effector_force(q, f_tip, Ms, Gs, ss)
    mass = mass_matrix(q, Ms, Gs, ss)
    return np.linalg.inv(mass) @ acc_tor


def euler_step(q, dq, ddq, dt):
    # 一阶欧拉数值积分
    return q + dq * dt, dq + ddq * dt


def id_trajectory(qs, dqs, ddqs, g, f_tips, Ms, Gs, ss):
    # 用逆动力学估计轨迹力矩
    taus = np.zeros_like(qs)
    for i in range(qs.shape[0]):
        taus[i, :] = inverse_dynamics(qs[i, :], dqs[i, :],
                                      ddqs[i, :], g, f_tips[i, :], Ms, Gs, ss)
    return taus


def fd_trajectory(q, dq, taus, g, f_tips, Ms, Gs, ss, dt, int_res):
    # 用正动力学仿真机器人运动
    qs = np.zeros_like(taus)
    dqs = np.zeros_like(taus)
    qs[0, :] = np.array(q)
    dqs[0, :] = np.array(dq)

    for i in range(taus.shape[0] - 1):
        for _ in range(int_res):
            ddq = forward_dynamics(q, dq, taus[i, :], g,
                                   f_tips[i, :], Ms, Gs, ss)
            q, dq = euler_step(q, dq, ddq, 1.0 * dt / int_res)
        qs[i+1, :] = q
        dqs[i+1, :] = dq

    return qs, dqs


'''
trajectory planning
'''


def cubic_time_scaling(Tf, t):
    # 三次多项式时间标度
    return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3


def quintic_time_scaling(Tf, t):
    # 五次多项式时间标度
    return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
        + 6 * (1.0 * t / Tf) ** 5


def joint_trajectory(qs, qe, Tf, N, method):
    # 计算关节空间直线轨迹
    timegap = Tf / (N - 1.0)
    traj = np.zeros((N, len(qs)))
    for i in range(N):
        if method == 3:
            s = cubic_time_scaling(Tf, timegap * i)
        else:
            s = quintic_time_scaling(Tf, timegap * i)
        traj[i, :] = s * np.array(qe) + (1 - s) * np.array(qs)
    return traj


def cartesian_trajectory(xs, xe, Tf, N, method):
    # 计算笛卡尔空间直线轨迹
    timegap = Tf / (N - 1.0)
    traj = [[None]] * N
    Rstart, pstart = trans_to_Rp(xs)
    Rend, pend = trans_to_Rp(xe)
    Rv = matrix_log6(Rstart.T @ Rend)
    for i in range(N):
        if method == 3:
            s = cubic_time_scaling(Tf, timegap * i)
        else:
            s = quintic_time_scaling(Tf, timegap * i)
        R = Rstart @ matrix_exp3(Rv * s)
        p = s * np.array(pend) + (1 - s) * np.array(pstart)
        traj[i] = np.r_[np.c_[R, p], [0, 0, 0, 1]]
    return traj


def computed_torque(q, dq, e_int, g, Ms, Gs, ss,
                    qd, dqd, ddqd, Kp, Ki, Kd):
    # 计算力矩控制法
    err = qd - q
    ddqc = Kp * err + Ki * (e_int + err) + Kd * (dqd - dq) + ddqd
    torque = inverse_dynamics(q, dq, ddqc, g, np.zeros(6), Ms, Gs, ss)
    return torque


def simulate_control(q, dq, g, f_tips, Ms, Gs, ss,
                     qds, dqds, ddqds, g_est, M_est, G_est, Kp, Ki, Kd, dt, int_res):
    # 仿真计算力矩法控制
    m, n = qds.shape
    q_ = np.array(q)
    dq_ = np.array(dq)
    e_int = np.zeros(n)
    taus = np.zeros((m, n))
    qs = np.zeros((m, n))

    for i in range(m):

        tau = computed_torque(q_, dq_, e_int, g_est, M_est, G_est,
                              ss, qds[i, :], dqds[i, :], ddqds[i, :], Kp, Ki, Kd)

        for _ in range(int_res):
            ddq = forward_dynamics(q_, dq_, tau, g, f_tips[i, :], Ms, Gs, ss)
            q_, dq_ = euler_step(q_, dq_, ddq, 1.0 * dt / int_res)

        taus[i, :] = tau
        qs[i, :] = q_
        e_int += (qds[i, :] - q_) * dt

    return (taus, qs)


if __name__ == '__main__':
    pass
