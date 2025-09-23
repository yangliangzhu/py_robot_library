import numpy as np
import casadi as ca
from scipy.differentiate import jacobian
from .ik_type import *


class IkStandard:

    def __init__(self, model, name):
        self.model = model
        self.configure(name)

    def configure(self, name):
        self.name = name
        if name == IkType.IK_STANDARD:
            self.jac_func = self.model.jacobian
            self.diff_func = self.model.se3_diff
        elif name == IkType.IK_NORMAL:
            self.jac_func = self.model.jac_normal
            self.diff_func = self.model.normal_diff
        else:
            raise NotImplemented(f"[ik]{name} not implemented")

    def __call__(self, angle, target):
        tar_T = target
        # . for test gain = 1.0 -> hard to find solution
        max_iter = 20.0
        max_iter_sup = 50
        pre_T = self.model.fk(angle)
        t = 0
        diff = self.diff_func(pre_T, tar_T)
        dis1 = np.linalg.norm(diff[:3])
        dis2 = np.linalg.norm(diff[3:])
        eps = 1e-5
        err_norm = max(dis1, dis2)
        if err_norm < eps:
            return angle, True

        last_err_norm = 1e5
        step_size = 1.0
        pos_step_ = 1e-1
        rot_step_ = 5e-2
        qb = np.array(angle)

        while t < max_iter:
            if (err_norm > last_err_norm):
                factor = min(0.2, last_err_norm / err_norm)
                max_iter = t + 1 + (max_iter - t - 1) / factor
                max_iter = min(max_iter_sup, max_iter)
                step_size *= factor
                if (step_size < 1e-4):
                    # * ik fail
                    return angle, False
                angle = np.array(qb)
            else:
                qb = np.array(angle)
                jac = self.jac_func(angle)
                U, S, V = np.linalg.svd(jac)
                S[S < 0.01] = 0  # 这里相当于“传动比”小于1%就放弃这个方向
                S[S >= 0.01] = 1.0 / S[S >= 0.01]

                # . comment: 阈值设为0.05会丢失很多解
                dim_v = V.shape[0]
                dim_u = U.shape[0]
                S_ = np.zeros((dim_v, dim_u))
                for i in range(S.shape[0]):
                    S_[i, i] = S[i]
                inv_jac = V.T @ S_ @ U.T

                if (dis1 > pos_step_):
                    diff[:3] = diff[:3] * pos_step_ / dis1
                if (dis2 > rot_step_):
                    diff[3:] = diff[3:] * rot_step_ / dis2
                dq = inv_jac @ diff.flatten()

            angle = angle + dq * step_size
            angle = np.clip(angle, self.model.lower_bounds,
                            self.model.upper_bounds)
            pre_T = self.model.fk(angle)
            diff = self.diff_func(pre_T, tar_T)
            dis1 = np.linalg.norm(diff[:3])
            dis2 = np.linalg.norm(diff[3:])
            last_err_norm = err_norm
            err_norm = max(dis1, dis2)
            if err_norm < eps:
                # * ik success
                return angle, True
            t = t + 1
        # * ik fail
        return angle, False


class IkNullSpace:
    '''
    first optimize in null space for Er3 7-joint robot arm
    next: we can loose yaw, so enable 2-dimensional null space for Er3
        and 1-dimensional null space for Sr5
    '''

    def __init__(self, model, name):
        self.model = model
        self.set_bounds_flag = False
        self.configure(name)

    def configure(self, name):
        self.name = name
        if name == IkType.IK_NULL:
            self.jac_func = self.model.jacobian
            self.diff_func = self.model.se3_diff
        elif name == IkType.IK_NULL_NORMAL:
            self.jac_func = self.model.jac_normal
            self.diff_func = self.model.normal_diff
        else:
            raise NotImplemented(f"[ik]{name} not implemented")

    def set_bounds(self):
        # TODO: 设置最大速度 根据超出软限位的幅度计算百分比 得到最终使用的速度
        soft_limit = np.radians(20)
        self.soft_upper = self.model.upper_bounds - soft_limit
        self.soft_lower = self.model.lower_bounds + soft_limit
        if np.any(self.soft_upper - self.soft_lower < 0):
            raise ValueError("soft upper must be >= soft lower")
        self.set_bounds_flag = True

    def subtask_avoid_limit(self, q):
        velocity = np.radians(100)
        dt = 2e-2
        v = np.zeros_like(q)
        v_zero = v.copy()
        for i in range(self.model.num_dof):
            if q[i] > self.soft_upper[i]:
                v[i] = -1.0
            elif q[i] < self.soft_lower[i]:
                v[i] = 1.0
        if np.allclose(v, v_zero):
            return v_zero, False
        cmd = velocity * dt * v
        return cmd, True

    def sub_task_manip(self, q):
        velocity = np.radians(100)
        dt = 2e-2
        grad = self.model.derivative_manip(q)
        grad_vec = grad / (np.linalg.norm(grad) + 1e-10)
        cmd = velocity * dt * grad_vec
        return cmd

    def __call__(self, angle, target):
        if not self.set_bounds_flag:
            self.set_bounds()

        tar_T = target
        # . for test gain = 1.0 -> hard to find solution
        max_iter = 20.0
        max_iter_sup = 50
        pre_T = self.model.fk(angle)
        t = 0
        diff = self.diff_func(pre_T, tar_T)
        dis1 = np.linalg.norm(diff[:3])
        dis2 = np.linalg.norm(diff[3:])
        eps = 1e-5
        err_norm = max(dis1, dis2)
        if err_norm < eps:
            return angle, True

        last_err_norm = 1e5
        step_size = 1.0
        pos_step_ = 1e-1
        rot_step_ = 5e-2
        qb = np.array(angle)

        while t < max_iter:
            if (err_norm > last_err_norm):
                factor = min(0.2, last_err_norm / err_norm)
                max_iter = t + 1 + (max_iter - t - 1) / factor
                max_iter = min(max_iter_sup, max_iter)
                step_size *= factor
                if (step_size < 1e-4):
                    # * ik fail
                    return angle, False
                angle = np.array(qb)
            else:
                qb = np.array(angle)
                jac = self.jac_func(angle)
                U, S, V = np.linalg.svd(jac)
                dim_v = V.shape[0]
                dim_u = U.shape[0]
                S_null = np.eye(dim_v)
                singular_threshold = 0.01
                for i in range(S.shape[0]):
                    if S[i] < singular_threshold:
                        S[i] = 0.0
                    else:
                        S[i] = 1.0 / S[i]
                        S_null[i, i] = 0.0

                S_ = np.zeros((dim_v, dim_u))
                for i in range(S.shape[0]):
                    S_[i, i] = S[i]
                inv_jac = V.T @ S_ @ U.T
                null_jac = V.T @ S_null @ V

                null_space_exist = not np.allclose(
                    S_null, np.zeros_like(S_null))

                if (dis1 > pos_step_):
                    diff[:3] = diff[:3] * pos_step_ / dis1
                if (dis2 > rot_step_):
                    diff[3:] = diff[3:] * rot_step_ / dis2
                dq_null, activate_limit_task = self.subtask_avoid_limit(angle)
                dq = inv_jac @ diff.flatten()
                if activate_limit_task and null_space_exist:
                    dq += null_jac @ dq_null
                elif null_space_exist:
                    dq_null2 = self.sub_task_manip(angle)
                    dq += null_jac @ dq_null2

            angle = angle + dq * step_size
            angle = np.clip(angle, self.model.lower_bounds,
                            self.model.upper_bounds)
            pre_T = self.model.fk(angle)
            diff = self.diff_func(pre_T, tar_T)
            dis1 = np.linalg.norm(diff[:3])
            dis2 = np.linalg.norm(diff[3:])
            last_err_norm = err_norm
            err_norm = max(dis1, dis2)
            if err_norm < eps:
                # * ik success
                return angle, True
            t = t + 1
        # * ik fail
        return angle, False


class IkNaive:

    def __init__(self, model, name):
        self.model = model
        self.configure(name)

    def configure(self, name):
        self.name = name
        self.jac_func = self.model.jacobian
        self.diff_func = self.model.se3_diff

    def solution_converge(self, pre_T, tar_T):
        diff = self.diff_func(pre_T, tar_T)
        eps_trans = 1e-4
        eps_rot = np.deg2rad(0.5)
        trans_converge = np.linalg.norm(diff[:3]) < eps_trans
        rot_converge = np.linalg.norm(diff[3:]) < eps_rot
        converge = (trans_converge and rot_converge)
        return converge, diff

    def __call__(self, angle, target):
        tar_T = target
        pre_T = self.model.fk(angle)
        kSingular = 2e-10
        _, diff = self.solution_converge(pre_T, tar_T)

        jac = self.jac_func(angle)
        U, S, V = np.linalg.svd(jac)
        S[S < kSingular] = 0
        S[S >= kSingular] = 1.0 / S[S >= kSingular]

        dim_v = V.shape[0]
        dim_u = U.shape[0]
        S_ = np.zeros((dim_v, dim_u))
        for i in range(S.shape[0]):
            S_[i, i] = S[i]
        inv_jac = V.T @ S_ @ U.T

        dq = inv_jac @ diff.flatten()

        angle = angle + dq
        angle = np.clip(angle, self.model.lower_bounds,
                        self.model.upper_bounds)
        return angle, True


class IkQp:
    # TODO：one step ik-qp is still not available
    def __init__(self, model, name):
        self.model = model
        self.configure(name)
        self.last_δq = np.zeros(7)
        # HACK for debug
        self.ddq_max = np.ones(7) * 100.0
        self.dq_max = np.ones(7) * np.deg2rad(80)
        self.q_max = self.model.get_upper_bounds()
        self.q_min = self.model.get_lower_bounds()
        self.dt = 1e-3
        self.penalty_dq = 1e-4
        self.ct = 0

    def configure(self, name):
        self.name = name
        self.jac_func = self.model.jacobian
        self.diff_func = self.model.se3_diff

        H = ca.DM.zeros(7, 7)
        A = ca.DM.zeros(7, 7)
        qp = {
            'h': H.sparsity(),
            'a': A.sparsity(),
        }
        opts = {
            'printLevel': 'none',        # 关闭qpOASES输出
            'error_on_fail': False,
        }
        self.solver = ca.conic('solver', 'qpoases', qp, opts)
        self.initialized = False

    def get_qp_coefficients(self, jac, q_ref, dx_des, verbose=False):
        # build solver using casadi
        '''
          min ||J * δq - dx||^2 + w * ||δq - δq_last||^2
          s.t. -ddq * dt^2    < δq - δq_last < ddq * dt^2
                - dq_max * dt <      δq      < dq_max * dt
                        q_min <    q + δq    < q_max
        '''
        # TODO: add avoid joint limit
        H = jac.T @ jac + self.penalty_dq * np.eye(7)
        g = - (self.penalty_dq * q_ref + jac.T @ dx_des)
        # constraints
        lbx_acc = self.last_δq - self.ddq_max * self.dt**2
        ubx_acc = self.last_δq + self.ddq_max * self.dt**2
        lbx_vel = - self.dq_max * self.dt
        ubx_vel = self.dq_max * self.dt
        lbx_pos = self.q_min - q_ref
        ubx_pos = self.q_max - q_ref

        # details
        if verbose:
            print(f'lbx_acc: {lbx_acc}')
            print(f'ubx_acc: {ubx_acc}')
            print(f'lbx_vel: {lbx_vel}')
            print(f'ubx_vel: {ubx_vel}')
            print(f'lbx_pos: {lbx_pos}')
            print(f'ubx_pos: {ubx_pos}')

        # combined constraints
        lbx = np.maximum(lbx_acc, lbx_vel)
        lbx = np.maximum(lbx, lbx_pos)
        ubx = np.minimum(ubx_acc, ubx_vel)
        ubx = np.minimum(ubx, ubx_pos)

        return H, g, lbx, ubx

    def initial_call(self, q_ref, target):
        tar_T = target
        pre_T = self.model.fk(q_ref)
        dx_des = self.diff_func(pre_T, tar_T).flatten()
        jac = self.model.jacobian(q_ref)
        kSingular = 0.01
        U, S, V = np.linalg.svd(jac)
        S[S < kSingular] = 0  # 这里相当于“传动比”小于1%就放弃这个方向
        S[S >= kSingular] = 1.0 / S[S >= kSingular]

        # . comment: 阈值设为0.05会丢失很多解
        S_ = np.zeros_like(jac.T)
        for i in range(S.shape[0]):
            S_[i, i] = S[i]
        inv_jac = V.T @ S_ @ U.T
        opti_δq2 = inv_jac @ dx_des

        if not self.initialized:
            self.last_δq = opti_δq2
        self.initialized = True

        H, g, lbx, ubx = self.get_qp_coefficients(jac, q_ref, dx_des)
        res = self.solver(h=H, g=g, lbx=lbx, ubx=ubx)
        opti_δq = res['x'].full().flatten()
        cost_opt = 0.5 * np.inner(opti_δq, H @ opti_δq) + np.inner(opti_δq, g)
        cost_raw = 0.5 * np.inner(opti_δq2, H @ opti_δq2) + \
            np.inner(opti_δq2, g)
        # check condition
        if self.ct < 100:
            print("check lbx limits:", np.all(opti_δq2 > lbx))
            print("check ubx limits:", np.all(opti_δq2 < ubx))
            print("check cost raw:", cost_raw)
            print("check cost opt:", cost_opt)
            self.ct += 1
        return opti_δq2

    def __call__(self, q_ref, target):
        if not self.initialized:
            opti_δq = self.initial_call(q_ref, target)
            sol = q_ref + opti_δq
            return sol, True
        # else:
        #     opti_δq2 = self.initial_call(q_ref, target)
        tar_T = target
        pre_T = self.model.fk(q_ref)
        dx_des = self.diff_func(pre_T, tar_T).flatten()
        jac = self.model.jacobian(q_ref)
        H, g, lbx, ubx = self.get_qp_coefficients(jac, q_ref, dx_des)
        res = self.solver(h=H, g=g, lbx=lbx, ubx=ubx)
        opti_δq = res['x'].full().flatten()
        self.last_δq = opti_δq

        sol = q_ref + opti_δq
        return sol, True
