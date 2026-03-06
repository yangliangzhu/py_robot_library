import numpy as np

# todo rotx, roty, rotz


def rot_x(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])


def rot_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def rot_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def nervCartToAffine(p, rad=False):
    omega = np.array(p[3:])
    result = np.eye(4)
    if not rad:
        omega = np.deg2rad(omega)
    result[:3, 3] = p[:3] * 1e-3  # mm to m
    result[:3, :3] = rot_z(omega[2]) @ rot_y(omega[1]) @ rot_x(omega[0])
    return result


def nervRpy(p, rad=False):
    omega = np.array(p[3:])
    result = np.eye(3)
    if not rad:
        omega = np.deg2rad(omega)
    result[:3, :3] = rot_z(omega[2]) @ rot_y(omega[1]) @ rot_x(omega[0])
    return result


def rot2rpy(rot):
    """
    将旋转矩阵转换为RPY欧拉角 (Roll, Pitch, Yaw)
    旋转顺序: ZYX (先绕Z轴旋转yaw，再绕Y轴旋转pitch，最后绕X轴旋转roll)
    """
    # 提取旋转矩阵元素
    r11, r12, r13 = rot[0, :]
    r21, r22, r23 = rot[1, :]
    r31, r32, r33 = rot[2, :]

    # 计算欧拉角
    # 处理奇异情况（万向锁）
    if abs(r31) < 0.9999999:
        # 正常情况
        y = np.arctan2(r21, r11)  # yaw (绕Z轴)
        p = np.arctan2(-r31, np.sqrt(r32**2 + r33**2))  # pitch (绕Y轴)
        r = np.arctan2(r32, r33)  # roll (绕X轴)
    else:
        # 奇异情况（万向锁）
        y = 0  # 可以任意设置yaw角
        if r31 < 0:  # pitch = +90度
            p = np.pi / 2
            r = np.arctan2(r12, r13)
        else:  # pitch = -90度
            p = -np.pi / 2
            r = np.arctan2(-r12, -r13)

    return np.array([r, p, y])


def rpy2rot(rpy):
    return rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])


def rot2quat(rot):
    """
    Convert a rotation matrix to quaternion.

    Parameters:
    rot (numpy.ndarray): 3x3 rotation matrix

    Returns:
    numpy.ndarray: quaternion [w, x, y, z]
    """
    # Ensure the input is a 3x3 matrix
    rot = np.asarray(rot)
    if rot.shape != (3, 3):
        raise ValueError("Input must be a 3x3 rotation matrix")

    # Calculate quaternion components
    trace = np.trace(rot)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif (rot[0, 0] > rot[1, 1]) and (rot[0, 0] > rot[2, 2]):
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s

    return np.array([qw, qx, qy, qz])


def quat2rot(q):
    """
    Convert a quaternion to rotation matrix.

    Parameters:
    q (numpy.ndarray): quaternion [w, x, y, z]

    Returns:
    numpy.ndarray: 3x3 rotation matrix
    """
    # Normalize the quaternion to ensure it's a unit quaternion
    q = np.asarray(q)
    if q.shape != (4,):
        raise ValueError(
            "Input must be a quaternion with 4 elements [w, x, y, z]")

    # Normalize quaternion
    q = q / np.linalg.norm(q)

    w, x, y, z = q

    # Calculate rotation matrix elements
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    yy = y * y
    yz = y * z
    yw = y * w
    zz = z * z
    zw = z * w

    # Build rotation matrix
    rot = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - zw),     2 * (xz + yw)],
        [2 * (xy + zw), 1 - 2 * (xx + zz),     2 * (yz - xw)],
        [2 * (xz - yw),     2 * (yz + xw), 1 - 2 * (xx + yy)]
    ])

    return rot
