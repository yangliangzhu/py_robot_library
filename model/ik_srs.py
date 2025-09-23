import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt


class RobotIK:
    def __init__(self, d_bs=0.3415, d_se=0.394, d_ew=0.366, d_wt=0.2503):
        self.d_bs = d_bs
        self.d_se = d_se
        self.d_ew = d_ew
        self.d_wt = d_wt
        self.len_0_bs = np.array([0, 0, d_bs])
        self.len_3_se = np.array([0, -d_se, 0])
        self.len_4_ew = np.array([0, 0, d_ew])
        self.len_7_wt = np.array([0, 0, d_wt])
        self.alpha = np.array([-1, 1, -1, 1, -1, 1, 0]) * np.pi / 2


    def skew_vector(self, v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def rotation_axis(self, theta, index_joint):
        ca = cos(self.alpha[index_joint - 1])
        sa = sin(self.alpha[index_joint - 1])
        return np.array([[cos(theta), -sin(theta) * ca, sin(theta) * sa],
                         [sin(theta), cos(theta) * ca, -cos(theta) * sa],
                         [0, sa, ca]])

    def init_shoulder_joint(self, x, y):
        alpha = np.arctan2(x[2], x[0])
        beta = np.arctan2(y[1], y[0])
        amplitude1 = np.sqrt(x[0]**2 + x[2]**2)
        amplitude2 = np.sqrt(y[0]**2 + y[1]**2)
        theta_02 = np.arcsin(-y[2] / amplitude1) + alpha
        theta_01 = np.arcsin(-x[1] / amplitude2) + beta

        if np.cos(theta_01 - alpha) * np.cos(theta_02 - beta) < 0:
            theta_02 = np.pi - theta_02

        return theta_01, theta_02

    def extract_r_x_from_matrix(self, mat):
        """从4x4矩阵中提取旋转部分和位移部分"""
        r = mat[:3, :3]
        x = mat[:3, 3]
        return r, x

    def inverse(self, mat, phi):
        """从4x4矩阵中提取r和x,并计算逆运动学"""
        r, x = self.extract_r_x_from_matrix(mat)
        phi = np.radians(phi)
        x_0_sw = x - self.len_0_bs - r @ self.len_7_wt
        cos_theta_4 = (np.linalg.norm(x_0_sw)**2 - self.d_se**2 - self.d_ew**2) / (2 * self.d_se * self.d_ew)
        theta_4 = np.arccos(cos_theta_4)
        u_0_sw = x_0_sw / np.linalg.norm(x_0_sw)

        x_for_calculation = self.rotation_axis(0, 3) @ (self.len_3_se + self.rotation_axis(theta_4, 4) @ self.len_4_ew)
        y_for_calculation = x_0_sw
        theta_1_ref, theta_2_ref = self.init_shoulder_joint(x_for_calculation, y_for_calculation)
        r_03_ref = self.rotation_axis(theta_1_ref, 1) @ self.rotation_axis(theta_2_ref, 2) @ self.rotation_axis(0, 3)

        cross_matrix_sw = self.skew_vector(u_0_sw)
        A_s = cross_matrix_sw @ r_03_ref
        B_s = -cross_matrix_sw @ cross_matrix_sw @ r_03_ref
        C_s = np.array(np.matrix(u_0_sw).T @ np.matrix(u_0_sw)) @ r_03_ref
        r_03 = A_s * sin(phi) + B_s * cos(phi) + C_s

        theta_1 = np.arctan2(-r_03[1, 1], -r_03[0, 1])
        theta_2 = np.arccos(-r_03[2, 1])
        theta_3 = np.arctan2(r_03[2, 2], -r_03[2, 0])

        A_w = self.rotation_axis(theta_4, 4).T @ A_s.T @ r
        B_w = self.rotation_axis(theta_4, 4).T @ B_s.T @ r
        C_w = self.rotation_axis(theta_4, 4).T @ C_s.T @ r
        r_47 = A_w * sin(phi) + B_w * cos(phi) + C_w

        theta_5 = np.arctan2(r_47[1, 2], r_47[0, 2])
        theta_6 = np.arccos(r_47[2, 2])
        theta_7 = np.arctan2(r_47[2, 1], -r_47[2, 0])

        return np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7])


    def forward(self, ang):
        Joint1 = self.mdh_mat(ang[0], self.d_bs, 0, 0)
        Joint2 = self.mdh_mat(ang[1], 0, 0, -np.pi / 2)
        Joint3 = self.mdh_mat(ang[2], self.d_se, 0, np.pi / 2)
        Joint4 = self.mdh_mat(ang[3], 0, 0, -np.pi / 2)
        Joint5 = self.mdh_mat(ang[4], self.d_ew, 0, np.pi / 2)
        Joint6 = self.mdh_mat(ang[5], 0, 0, -np.pi / 2)
        Joint7 = self.mdh_mat(ang[6], self.d_wt, 0, np.pi / 2)
        return Joint1 @ Joint2 @ Joint3 @ Joint4 @ Joint5 @ Joint6 @ Joint7

    def mdh_mat(self, theta, d, a, alpha):
        return np.array([
            [cos(theta), -sin(theta), 0, a],
            [cos(alpha) * sin(theta), cos(alpha) * cos(theta), -sin(alpha), -d * sin(alpha)],
            [sin(alpha) * sin(theta), sin(alpha) * cos(theta), cos(alpha), d * cos(alpha)],
            [0, 0, 0, 1]
        ])

    def show_error(self, result, mat):
        r_07_d, x_07_d = self.extract_r_x_from_matrix(mat)  # 从矩阵中提取旋转和位移部分
        error_rotation = result[:3, :3] @ np.linalg.inv(r_07_d) - np.eye(3)
        error_translation = result[:3, 3] - x_07_d
        print('error of r: ', np.linalg.norm(error_rotation))
        print('error of x: ', np.linalg.norm(error_translation))

    def compute_ref_vec(self, mat):
        sol = self.inverse(mat, 0)
        np.set_printoptions(precision=3, suppress=True)
        # print(sol)
        # 正向运动学计算关键点位置
        T01 = self.mdh_mat(sol[0], self.d_bs, 0, 0)
        T12 = self.mdh_mat(sol[1], 0, 0, -np.pi/2)
        T23 = self.mdh_mat(sol[2], self.d_se, 0, np.pi/2)
        T34 = self.mdh_mat(sol[3], 0, 0, -np.pi/2)
        T45 = self.mdh_mat(sol[4], self.d_ew, 0, np.pi/2)
        
        # 计算关键点坐标系
        T03 = T01 @ T12 @ T23  # 到肘关节的变换矩阵
        T04 = T03 @ T34 @ T45

        p1 = T01[:3, 3]
        p3 = T03[:3, 3]
        p4 = T04[:3, 3]        
        
        # 计算法向量
        vec1 = p3 - p1
        vec2 = p4 - p3
        # 计算夹角
        plane_norm = np.cross(vec1, vec2)       
        return plane_norm / np.linalg.norm(plane_norm)

    def compute_arm_phi(self, joint_angles):
        """
        根据关节角计算当前臂角φ
        参数:
            joint_angles: 7维关节角数组 (弧度制)
        返回:
            phi: 当前臂角 (角度制)
        """
        # 正向运动学计算关键点位置
        T01 = self.mdh_mat(joint_angles[0], self.d_bs, 0, 0)
        T12 = self.mdh_mat(joint_angles[1], 0, 0, -np.pi/2)
        T23 = self.mdh_mat(joint_angles[2], self.d_se, 0, np.pi/2)
        T34 = self.mdh_mat(joint_angles[3], 0, 0, -np.pi/2)
        T45 = self.mdh_mat(joint_angles[4], self.d_ew, 0, np.pi/2)
        
        # 计算关键点坐标系
        T03 = T01 @ T12 @ T23  # 到肘关节的变换矩阵
        T04 = T03 @ T34 @ T45

        p1 = T01[:3, 3]
        p3 = T03[:3, 3]
        p4 = T04[:3, 3]        
        
        # 计算法向量
        vec1 = p3 - p1
        vec2 = p4 - p3
        # 计算夹角
        plane_norm = np.cross(vec1, vec2)
        plane_norm = plane_norm / np.linalg.norm(plane_norm)
        pos = self.forward(joint_angles)       
        ref_norm = self.compute_ref_vec(pos)
        # ref_norm = np.array([0, 1, 0])
        # print(ref_norm)

        cos_angle = np.inner(plane_norm, ref_norm)
        angle = np.arccos(cos_angle)

        common_norm = np.cross(plane_norm, ref_norm)
        # print(common_norm / np.linalg.norm(common_norm))

        # print(plane_norm)
        return np.degrees(angle)


if __name__ == '__main__':
    robot = RobotIK()
    # q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    q = np.random.rand(7) * np.pi
    pos = robot.forward(q)

    # 使用4x4矩阵作为输入
    mat = pos
    r_07_d, x_07_d = robot.extract_r_x_from_matrix(mat)

    print(f"x = {x_07_d}")

    phi_set = []
    joints = []
    est_phi_set = []

    for phi in range(-180, 180, 5):
        try:
            angle = robot.inverse(mat, phi)
            result = robot.forward(angle)
            joints.append(angle)
            phi_set.append(phi)
            est_phi = robot.compute_arm_phi(angle)
            est_phi_set.append(est_phi)
        except Exception as e:
            print(f'degree {phi} is not solvable')

    joints = np.array(joints)
    for i in range(7):
        for j in range(len(phi_set) - 1):
            diff = joints[j + 1, i] - joints[j, i]
            if diff > 1.8 * np.pi:
                joints[j + 1, i] -= 2 * np.pi
            elif diff < -1.8 * np.pi:
                joints[j + 1, i] += 2 * np.pi

    # 臂角计算
    for i in range(7):
        plt.plot(phi_set, np.degrees(joints[:, i]), label=f'joint {i + 1}')
    plt.xlabel("phi-angle(free dof)")
    plt.ylabel("joint-angle(multiple solution)")
    plt.legend()
    plt.show()

    plt.plot(phi_set, est_phi_set, label='est_phi')
    plt.legend()
    plt.show()

