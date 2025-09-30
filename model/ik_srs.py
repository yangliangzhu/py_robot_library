import numpy as np
from numpy import cos, sin
from enum import Enum

"""
M. Shimizu, H. Kakuya, W. -K. Yoon, K. Kitagaki and K. Kosuge,
"Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators
With Joint Limits and Its Application to Redundancy Resolution,"
in IEEE Transactions on Robotics, vol. 24, no. 5, pp. 1131-1142, Oct. 2008,
doi: 10.1109/TRO.2008.2003266.


paper frame index: 0 ~ 6 + 7
code  frame index: 1 ~ 6 + Tool
Here for example, after q4 rotation, Frame 4 -> Frame 4 * Rz(q4) * Tz(d4) * Rx(alpha 4) * Tx(a4) ->
update Frame5
Frame {i} means when q1 != 0, ...., q{i-1}!=0, but q{i} = 0, the position of the Frame
"""


class FrameId(Enum):
    """
    关节编号枚举类
    """
    kFrame1 = 0
    kFrame2 = 1
    kFrame3 = 2
    kFrame4 = 3
    kFrame5 = 4
    kFrame6 = 5
    kFrame7 = 6
    kTool = 7

    def next(self):
        """
        获取下一个枚举值
        """
        next_value = self.value + 1
        # 检查下一个值是否在有效范围内
        if next_value <= FrameId.kFrame7.value:
            return FrameId(next_value)
        else:
            return None


class RobotSRS:
    """
    机器人逆运动学解析解计算器

    用于计算7自由度机械臂的逆运动学解析解，支持冗余自由度的臂角优化
    """

    def __init__(self, d_bs=0.3415, d_se=0.394, d_ew=0.366, d_wt=0.2503):
        """
        初始化机器人参数

        Parameters:
            d_bs (float): 基座到肩部的距离
            d_se (float): 肩部到肘部的距离
            d_ew (float): 肘部到腕部的距离
            d_wt (float): 腕部到末端执行器的距离
        """
        self.d_bs = d_bs
        self.d_se = d_se
        self.d_ew = d_ew
        self.d_wt = d_wt
        self.l_bs = np.array([0, 0, d_bs])
        self.l_se = np.array([0, -d_se, 0])
        self.l_ew = np.array([0, 0, d_ew])
        self.l_wt = np.array([0, 0, d_wt])
        self.dh_alpha = np.array([-1, 1, -1, 1, -1, 1, 0]) * np.pi / 2

    @staticmethod
    def cross(v):
        """
        计算向量的反对称矩阵

        Parameters:
            v (np.array): 3维向量

        Returns:
            np.array: 3x3反对称矩阵
        """
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

    def rot(self, theta: float, idx: FrameId):
        # TODO: rot from id1 to id2
        """
        计算相邻两个关节之间的旋转矩阵

        Parameters:
            theta (float): 旋转角度
            idx (FrameId): 关节索引 (1-7)

        Returns:
            np.array: 3x3旋转矩阵
        """

        alpha = self.dh_alpha[idx.value]
        ca = cos(alpha)
        sa = sin(alpha)
        ct = cos(theta)
        st = sin(theta)
        # print(ca, sa, ct, st)
        # * rotation part: Rz(theta) * Rx(alpha)
        return np.array([[ct, -st * ca,  st * sa],
                         [st,  ct * ca, -ct * sa],
                         [0.,       sa,       ca]])

    def rot_seq(self, q: np.ndarray, start_id: FrameId, end_id: FrameId):
        """
        计算相邻两个关节之间的旋转矩阵

        Parameters:
            theta (float): 旋转角度向量
            idx (FrameId): 关节索引 (1-7)

        Returns:
            np.array: 3x3旋转矩阵
        """
        mat = np.eye(3)
        if start_id.value >= end_id.value:
            return mat

        loop_id = start_id
        while loop_id.value < end_id.value:
            theta = q[loop_id.value]
            mat = mat @ self.rot(theta, loop_id)
            loop_id = loop_id.next()
            if loop_id == None:
                break
        return mat

    def init_shoulder_joint(self, rhs, lhs):
        """
        初始化肩部关节角度

        Parameters:
            rhs (np.array): x向量
            y (np.array): y向量

        Returns:
            tuple: (q01, q02) 关节角度
        """
        # * The following can be derived from R1.T * lhs = R2 * rhs
        # * as for multiple solution, i.e. arcsin(q) has two choices, the choice can be arbitrary

        alpha = np.arctan2(rhs[2], rhs[0])
        beta = np.arctan2(lhs[1], lhs[0])
        amplitude1 = np.sqrt(rhs[0]**2 + rhs[2]**2)
        amplitude2 = np.sqrt(lhs[0]**2 + lhs[1]**2)
        q02 = np.arcsin(-lhs[2] / amplitude1) + alpha
        q01 = np.arcsin(-rhs[1] / amplitude2) + beta

        return q01, q02

    def ik(self, mat, phi):
        """
        计算逆运动学解析解

        Parameters:
            mat (np.array): 4x4目标位姿矩阵
            phi (float): 臂角参数(单位: 度)

        Returns:
            np.array: 7个关节角度
        """
        phi = np.deg2rad(phi)

        R_1t = mat[:3, :3]
        p_1t = mat[:3, 3]

        # * get q4 acoording to geometric relationship
        # actually use R_17, but R_17 == R_1t
        p_sw = p_1t - self.l_bs - R_1t @ self.l_wt
        cos_q4 = (np.sum(np.square(p_sw)) - self.d_se **
                  2 - self.d_ew**2) / (2 * self.d_se * self.d_ew)
        if abs(cos_q4) > 1.0:
            return None, False
        q4 = np.arccos(cos_q4)
        u_sw = p_sw / np.linalg.norm(p_sw)

        # * reference plane related calculation: defined by q3 = 0
        # * equation: lhs = R1(q1) * R2(q2) * rhs
        rhs = self.rot(
            0, FrameId.kFrame3) @ (self.l_se + self.rot(q4, FrameId.kFrame4) @ self.l_ew)
        lhs = p_sw
        q1_ref, q2_ref = self.init_shoulder_joint(
            rhs, lhs)
        R_14_ref = self.rot_seq([q1_ref, q2_ref, 0],
                                FrameId.kFrame1, FrameId.kFrame4)

        # * once reference plane is obtained, q1~q3 is directly computed
        cross_sw = self.cross(u_sw)
        A_s = cross_sw @ R_14_ref
        B_s = -cross_sw @ cross_sw @ R_14_ref
        C_s = np.outer(u_sw, u_sw) @ R_14_ref
        R_14 = A_s * sin(phi) + B_s * cos(phi) + C_s

        if abs(R_14[2, 1]) > 1.0:
            return None, False

        q1 = np.arctan2(-R_14[1, 1], -R_14[0, 1])
        q2 = np.arccos(-R_14[2, 1])
        q3 = np.arctan2(R_14[2, 2], -R_14[2, 0])

        # * also for q5 ~ q7
        # * 这里为了简化计算合并了一些计算并采用了论文中矩阵的转置
        R_t1 = R_1t.T
        R_45 = self.rot(q4, FrameId.kFrame4)
        R_t5 = R_t1 @ R_14 @ R_45

        if abs(R_t5[2, 2]) > 1.0:
            return None, False

        q5 = np.arctan2(R_t5[2, 1], R_t5[2, 0])
        q6 = np.arccos(R_t5[2, 2])
        q7 = np.arctan2(R_t5[1, 2], -R_t5[0, 2])

        return np.array([q1, q2, q3, q4, q5, q6, q7]), True

    def get_coeffs_theta(self, mat):
        """
        给定末端位姿后计算theta相关的三角有理函数系数

        Parameters:
            mat (np.array): 4x4目标位姿矩阵

        Returns:
            dict: theta相关的三角有理函数系数
        """

        R_1t = mat[:3, :3]
        p_1t = mat[:3, 3]

        # * get q4 acoording to geometric relationship
        # actually use R_17, but R_17 == R_1t
        p_sw = p_1t - self.l_bs - R_1t @ self.l_wt
        cos_q4 = (np.sum(np.square(p_sw)) - self.d_se **
                  2 - self.d_ew**2) / (2 * self.d_se * self.d_ew)
        if abs(cos_q4) > 1.0:
            return None
        q4 = np.arccos(cos_q4)
        u_sw = p_sw / np.linalg.norm(p_sw)

        # * reference plane related calculation: defined by q3 = 0
        # * equation: lhs = R1(q1) * R2(q2) * rhs
        rhs = self.rot(
            0, FrameId.kFrame3) @ (self.l_se + self.rot(q4, FrameId.kFrame4) @ self.l_ew)
        lhs = p_sw
        q1_ref, q2_ref = self.init_shoulder_joint(
            rhs, lhs)
        R_14_ref = self.rot_seq([q1_ref, q2_ref, 0],
                                FrameId.kFrame1, FrameId.kFrame4)

        # * once reference plane is obtained, q1~q3 infomation is complete
        cross_sw = self.cross(u_sw)
        A_s = cross_sw @ R_14_ref
        B_s = -cross_sw @ cross_sw @ R_14_ref
        C_s = np.outer(u_sw, u_sw) @ R_14_ref

        # * also for q5 ~ q7
        R_54 = self.rot(q4, FrameId.kFrame4).T
        A_w = R_54 @ A_s.T @ R_1t
        B_w = R_54 @ B_s.T @ R_1t
        C_w = R_54 @ C_s.T @ R_1t

        return [A_s, B_s, C_s, A_w, B_w, C_w]

    def _fk_keypoints(self, q):
        """
        fk of each keypoint
        """
        Rs = self.rot_seq(q, FrameId.kFrame1, FrameId.kFrame4)
        ps = self.l_bs

        pe = ps + Rs @ self.l_se
        Re = Rs @ self.rot_seq(q, FrameId.kFrame4, FrameId.kFrame5)

        pw = pe + Re @ self.l_ew
        Rw = Re @ self.rot_seq(q, FrameId.kFrame5, FrameId.kFrame7)

        pt = pw + Rw @ self.l_wt
        Rt = Rw @ self.rot_seq(q, FrameId.kFrame7, FrameId.kTool)
        return ps, pe, pw, pt, Rt

    def fk_detail(self, q):
        ps, pe, pw, pt, _ = self._fk_keypoints(q)
        return ps, pe, pw, pt

    def fk(self, q):
        """
        fk kinematics
        symbols: [s] for shoulder, [e] for elbow, [w] for wrist, [t] for tool
        """
        _, _, _, pt, Rt = self._fk_keypoints(q)
        res = np.r_[np.c_[Rt, pt.reshape(3, 1)], [[0, 0, 0, 1]]]
        return res

    def compute_arm_phi(self, q):
        mat = self.fk(q)

        # * get rotation axis
        R_1t = mat[:3, :3]
        p_1t = mat[:3, 3]

        p_sw = p_1t - self.l_bs - R_1t @ self.l_wt
        u_sw = p_sw / np.linalg.norm(p_sw)

        q_ref, success = self.ik(mat, 0)
        if not success:
            return None

        ps = self.l_bs
        Rs = self.rot_seq(q_ref, FrameId.kFrame1, FrameId.kFrame4)
        pe = ps + Rs @ self.l_se
        Re = Rs @ self.rot_seq(q_ref, FrameId.kFrame4, FrameId.kFrame5)
        pw = pe + Re @ self.l_ew
        ref_plane_vec = np.cross(pw - ps, pe - ps)
        ref_plane_vec_length = np.linalg.norm(ref_plane_vec)
        if (ref_plane_vec_length < 1e-5):
            # case: singularity
            return 0.0
        ref_plane_vec /= ref_plane_vec_length

        Rs = self.rot_seq(q, FrameId.kFrame1, FrameId.kFrame4)
        pe = ps + Rs @ self.l_se
        Re = Rs @ self.rot_seq(q, FrameId.kFrame4, FrameId.kFrame5)
        pw = pe + Re @ self.l_ew
        plane_norm = np.cross(pw - ps, pe - ps)
        plane_norm_length = np.linalg.norm(plane_norm)
        if (plane_norm_length < 1e-5):
            raise ValueError('Singularity but not handled')
        plane_norm /= plane_norm_length

        # * common norm to get sign of phi
        cross_vec = np.cross(ref_plane_vec, plane_norm)
        sign = np.sign(np.inner(cross_vec, u_sw))
        # * inner to get cos(phi)
        cos_phi = np.inner(ref_plane_vec, plane_norm)
        phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))
        return np.rad2deg(phi) * sign


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    robot = RobotSRS()
    # q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    q = np.random.rand(7) * np.pi
    pos = robot.fk(q)

    # 使用4x4矩阵作为输入
    mat = pos
    r_07_d = mat[:3, :3]
    x_07_d = mat[:3, 3]

    print(f"x = {x_07_d}")

    phi_set = []
    joints = []
    est_phi_set = []

    for phi in range(-180, 180, 1):
        angle, success = robot.ik(mat, phi)
        if not success:
            print(f'degree {phi} is not solvable')
            continue
        result = robot.fk(angle)
        joints.append(angle)
        phi_set.append(phi)
        est_phi = robot.compute_arm_phi(angle)
        est_phi_set.append(est_phi)

    joints = np.array(joints)
    for i in range(7):
        for j in range(len(phi_set) - 1):
            diff = joints[j + 1, i] - joints[j, i]
            # rmk: here hardcode 1.5 pi is for catch q jump 2*pi
            if diff > 1.0 * np.pi:
                remove_ct = round(diff / (1.99 * np.pi))
                joints[j + 1, i] -= 2 * np.pi * remove_ct
            elif diff < -1.0 * np.pi:
                remove_ct = round(-diff / (1.99 * np.pi))
                joints[j + 1, i] += 2 * np.pi * remove_ct

    # 臂角计算
    for i in range(7):
        plt.plot(phi_set, np.degrees(joints[:, i]), label=f'joint {i + 1}')
    plt.xlabel("phi-angle(free dof)")
    plt.ylabel("joint-angle(multiple solution)")
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(phi_set, est_phi_set, label='est_phi')
    plt.legend()
    plt.grid()
    plt.show()
