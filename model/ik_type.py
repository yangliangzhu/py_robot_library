from enum import Enum


class IkType(Enum):

    IK_STANDARD = "ik standard"
    IK_NULL = "ik null"
    IK_NORMAL = "ik normal"
    IK_NULL_NORMAL = "ik null normal"
    IK_NAIVE = "ik naive"
    IK_QP = "ik qp"
