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
