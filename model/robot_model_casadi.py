import numpy as np
import casadi as ca
from .dh_param import *
from .ik_solver import *
from .ik_type import *


class RobotModelCasadi():
    ''' 
    matrix: 机械臂本体参数 matrix格式
    dh: 机械臂本体参数 dh格式
    base: 外置底座
    ee: 固定的连接在法兰上的工具trans
    tool: 可变工具trans
    '''

    def __init__(self):
        # casadi functions
        self.__fk = None
        self.__jacobian = None
        self.__pinv_jac = None
        self.__manip = None
        self.__d_manip = None
        self.build_se3_diff_functions()

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
        param = config["param"]
        # * parse robot params
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
        self.build_kinematics_functions()

    ########################## generate casadi expressions ############################
    @staticmethod
    def rot_z(theta):
        matrix = ca.SX_eye(4)
        matrix[0, 0] = ca.cos(theta)
        matrix[0, 1] = -ca.sin(theta)
        matrix[1, 0] = ca.sin(theta)
        matrix[1, 1] = ca.cos(theta)
        return matrix

    def fk_sym(self, joints):
        '''
        return the position of the end-effector.
        Unit: rad
        '''
        fk_list = []
        last_T = self.Ms[0]
        for i in range(self.num_dof + 1):
            if i < self.num_dof:
                theta = joints[i]
            else:
                theta = 0
            frame_transform = self.Ms[i + 1]
            rot_axis = self.rot_z(theta)

            T = last_T @ frame_transform @ rot_axis
            last_T = T
            fk_list.append(T)
        return fk_list[-1]

    def build_kinematics_functions(self):
        # generate fk
        q = ca.SX.sym('q', self.num_dof)
        last_frame = self.fk_sym(q)
        self.__fk = ca.Function('forward', [q], [last_frame])

        # using lie algebra
        rhsx = last_frame[:3, 3]
        jacx = ca.jacobian(rhsx, q)

        row1 = last_frame[0, :3]
        row2 = last_frame[1, :3]
        row3 = last_frame[2, :3]

        omegax = row2 @ ca.jacobian(row3, q)
        omegay = row3 @ ca.jacobian(row1, q)
        omegaz = row1 @ ca.jacobian(row2, q)

        rhs = ca.vertcat(jacx, omegax, omegay, omegaz)
        self.__jacobian = ca.Function('f', [q], [rhs])

        virtual_link = 0.15
        norm_z = last_frame[:3, 2] * virtual_link
        jac_norm_z = ca.jacobian(norm_z, q)
        rhs_normz = ca.vertcat(jacx, jac_norm_z)
        self.__jac_norm = ca.Function('f', [q], [rhs_normz])

        rhs_hess = ca.jacobian(rhs, q)
        self.__hessian = ca.Function('f', [q], [rhs_hess])

        rhsp = ca.pinv(rhs)
        self.__pinv_jac = ca.Function('fp', [q], [rhsp])

        manip = ca.sqrt(ca.det(rhs @ rhs.T))
        self.__manip = ca.Function('mp', [q], [manip])

        diff_manip = ca.jacobian(manip, q)
        self.__d_manip = ca.Function('dmp', [q], [diff_manip])

        # extended function, loose yaw
        # * method 1: virtual last link
        self.virtual_link = 0.15
        normal_vec = last_frame[:3, 2] * self.virtual_link
        jac_normal = ca.jacobian(normal_vec, q)
        rhs_normal = ca.vertcat(jacx, jac_normal)
        self.__jac_norm = ca.Function('jac_normal', [q], [rhs_normal])

        # * method 2: jacobian in tool frame and remove last row(Rz)
        return

    def build_se3_diff_functions(self):
        A = ca.SX.sym('A', 4, 4)
        B = ca.SX.sym('B', 4, 4)
        dx = (B - A)[0:3, 3]
        dr = B[0:3, 0:3] @ A[0:3, 0:3].T

        phi = ca.acos((dr[0, 0] + dr[1, 1] + dr[2, 2] - 1) / 2)
        omega = phi / (2 * np.sin(phi)) * (dr - dr.T)
        w1 = omega[2, 1]
        w2 = omega[0, 2]
        w3 = omega[1, 0]
        rhs = ca.vertcat(dx[0], dx[1], dx[2], w1, w2, w3)
        self.__se3_diff = ca.Function('diff_matrix', [A, B], [rhs])
        rhs_singular = ca.vertcat(dx[0], dx[1], dx[2], 0, 0, 0)
        self.__se3_diff_singular = ca.Function(
            'diff_matrix_sin', [A, B], [rhs_singular])

    ########################## user intercaces #############################
    def jacobian(self, angle):
        return np.array(self.__jacobian(angle))

    def pinv_jac(self, angle):
        return np.array(self.__pinv_jac(angle))

    def jac_normal(self, angle):
        return np.array(self.__jac_norm(angle))

    def hessian(self, angle):
        return np.array(self.__hessian(angle))

    def fk(self, angle):
        return np.array(self.__fk(angle))

    def se3_diff(self, A, B):
        if np.allclose(A[:3, :3], B[:3, :3], atol=1e-5):
            return np.array(self.__se3_diff_singular(A, B))
        else:
            return np.array(self.__se3_diff(A, B))

    def normal_diff(self, A, B):
        # used in ik norm
        diff = np.zeros(6)
        diff[:3] = (B - A)[:3, 3]
        diff[3:] = (B - A)[:3, 2] * self.virtual_link
        return diff

    def is_not_singular(self, q):
        res = np.abs(self.__manip(q))[0, 0]
        bigger = (res > 0.005)
        return bigger

    def manip(self, q):
        return np.array(self.__manip(q))[0, 0]

    def derivative_manip(self, q):
        return np.array(self.__d_manip(q)).flatten()

    def ik(self, angle, target):
        return self.ik_solver(angle, target)


if __name__ == '__main__':
    '''
    用例
    '''
    pass
