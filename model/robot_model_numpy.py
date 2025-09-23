import numpy as np
import casadi as ca
from .dh_param import *
from .ik_solver import *
from .ik_type import *


class RobotModelNumpy():
    ''' 
    matrix: 机械臂本体参数 matrix格式
    dh: 机械臂本体参数 dh格式
    base: 外置底座
    ee: 固定的连接在法兰上的工具trans
    tool: 可变工具trans
    '''

    def __init__(self):
        self.virtual_link = 0.15

    def ik_factory_method(self, solver_type):
        if solver_type == IkType.IK_STANDARD:
            self.ik_solver = IkStandard(self, solver_type)
        elif solver_type == IkType.IK_NORMAL:
            self.ik_solver = IkStandard(self, solver_type)
        elif solver_type == IkType.IK_NULL:
            self.ik_solver = IkNullSpace(self, solver_type)
        elif solver_type == IkType.IK_NULL_NORMAL:
            self.ik_solver = IkNullSpace(self, solver_type)
        elif solver_type == IkType.IK_NAIVE:
            self.ik_solver = IkNaive(self, solver_type)
        elif solver_type == IkType.IK_QP:
            self.ik_solver = IkQp(self, solver_type)

    ########################## configure robot ########################################

    def build(self, type, solver_type, config, base=np.eye(4), ee=np.eye(4), tool=np.eye(4)):

        # * parse robot params
        param = config["param"]
        if type == 'dh':
            self.num_dof = len(param.dh_list)
            self.Ms = get_matrix_list(param, base, ee)
        elif type == 'mat':
            self.num_dof = len(param)
            self.Ms = [base]
            self.Ms.extend(param)
            self.Ms.append(ee)
        else:
            raise NotImplementedError("param type not supported")

        # * set tool and generate kinematics
        self.fixed_offset = self.Ms[-1].copy()  # 存下来，后续不再更改
        self.set_tool(tool)
        self.set_bounds(config['lower'], config['upper'])

        # * this should be done last
        self.ik_factory_method(solver_type)

    def get_upper_bounds(self):
        return self.upper_bounds

    def get_lower_bounds(self):
        return self.lower_bounds

    def set_bounds(self, lower_bounds=None, upper_bounds=None):
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds

    def set_tool(self, matrix):
        ''' 更新工具偏移：需要更新总偏移并重新计算符号表达式  '''
        self.Ms[-1] = self.fixed_offset @ matrix

    ########################## generate casadi expressions ############################
    @staticmethod
    def rot_z(theta):
        matrix = np.eye(4)
        matrix[0, 0] = np.cos(theta)
        matrix[0, 1] = -np.sin(theta)
        matrix[1, 0] = np.sin(theta)
        matrix[1, 1] = np.cos(theta)
        return matrix

    def fk(self, q):
        '''
        return the position of the end-effector.
        Unit: rad
        '''
        T = self.Ms[0]
        for i in range(self.num_dof):
            frame_transform = self.Ms[i + 1]
            rot_axis = self.rot_z(q[i])
            T = T @ frame_transform @ rot_axis
        T = T @ self.Ms[self.num_dof + 1]
        # print(self.Ms)
        return T

    def se3_diff(self, A, B):
        dx = (B - A)[0:3, 3]
        dr = B[0:3, 0:3] @ A[0:3, 0:3].T
        if np.allclose(A[:3, :3], B[:3, :3], atol=1e-5):
            w1, w2, w3 = 0, 0, 0
        else:
            phi = np.arccos((dr[0, 0] + dr[1, 1] + dr[2, 2] - 1) / 2)
            omega = phi / (2 * np.sin(phi)) * (dr - dr.T)
            w1 = omega[2, 1]
            w2 = omega[0, 2]
            w3 = omega[1, 0]
        res = np.array([dx[0], dx[1], dx[2], w1, w2, w3]).reshape(6, 1)
        return res

    def normal_diff(self, A, B):
        # used in ik norm
        diff = np.zeros(6)
        diff[:3] = (B - A)[:3, 3]
        diff[3:] = (B - A)[:3, 2] * self.virtual_link
        return diff

    ########################## user intercaces #############################
    def jacobian(self, q):
        '''
        return the position of the end-effector.
        Unit: rad
        '''
        frame_list = []
        pose = self.Ms[0]
        for i in range(self.num_dof):
            frame_transform = self.Ms[i + 1]
            rot_axis = self.rot_z(q[i])
            pose = pose @ frame_transform @ rot_axis
            frame_list.append(pose.copy())
        tool_pose = pose @ self.Ms[-1]

        jac = np.zeros((6, self.num_dof))
        for i in range(self.num_dof):
            offset = tool_pose[:3, 3] - frame_list[i][:3, 3]
            angular_v = frame_list[i][:3, 2]
            cart_v = np.cross(angular_v, offset)
            jac[:3, i] = cart_v
            jac[3:, i] = angular_v
        return jac

    def jac_normal(self, q):
        jac = self.jacobian(q)
        z_axis = self.fk(q)[:3, 2]
        z_axis_hat = np.array([[0., -z_axis[2], z_axis[1]],
                               [z_axis[2], 0., -z_axis[0]],
                               [-z_axis[1], z_axis[0], 0.]])
        virtual_link = 0.15
        jac[3:, :] = - z_axis_hat @ jac[3:, :] * virtual_link
        return jac

    def is_not_singular(self, q):
        res = np.abs(self.manip(q))
        bigger = (res > 0.005)
        return bigger

    def manip(self, q):
        jac = self.jacobian(q)
        return np.linalg.det(jac @ jac.T)

    def hessian(self, q):
        # * numerical hessian
        result = np.zeros((6, self.num_dof, self.num_dof))
        eps = 1e-4
        jac_base = self.jacobian(q)
        for i in range(7):
            delta_q = np.zeros(7)
            delta_q[i] = eps
            jac = self.jacobian(q + delta_q)
            hess_est = (jac - jac_base) / eps
            result[:, :, i] = hess_est
        return result

    def derivative_manip(self, q):
        '''
            numerical derivative
            rmk: analytical calculation is slower
            since we don't need the result to be so accurate,
            wo choose to use numerical calculation
            ref: robot_model_casadi.ipynb for [Analytical calculation]
        '''

        grad_manip = np.zeros(7)
        delta = 1e-4
        jac = self.jacobian(q)
        manip_square = np.linalg.det(jac @ jac.T)
        manip = np.sqrt(manip_square)
        coeff = 1.0 / (2.0 * manip) if manip > delta else 0.0
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = delta
            jac_new = self.jacobian(q + dq)
            manip_square_new = np.linalg.det(jac_new @ jac_new.T)
            # * use f(q+dq) - f(q) instead of f(q+dq) - f(q-dq), for speed
            diff = (manip_square_new - manip_square) / delta
            grad_manip[i] = diff
        return grad_manip * coeff

    def ik(self, angle, target):
        return self.ik_solver(angle, target)


if __name__ == '__main__':
    '''
    用例
    '''
    pass
