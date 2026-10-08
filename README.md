# py_robot_library

Python 机器人运动学库：DH 参数、正逆运动学、Jacobian 与几何工具，NumPy 与 CasADi 两套后端可互换。

本库的 C++ 对应实现是 [robot-model-cpp](https://gitee.com/yangliangzhu_rob/robot-model-cpp)，两者实现同一套算法、互为交叉校验。Gitee 上的 README 讲的是这套**库**；本仓库（GitHub）是**逆解分支结构研究**的主场，库本身在这里只作配角。

## 逆解分支结构研究（`study/`）

> ### 🔎 本仓库的主线，是 6R 臂的逆解分支结构
>
> 研究在珞石 SR5 / SR0 和一台 3R 基准臂上展开，问的是同一件事：**一台 6R 臂的多组逆解彼此如何连通**，以及要把这种话讲成可测量的结论，需要哪些仪器。下面三个问题各有答案，过程中的每一次推翻都留在记录里。

### 问题一：哪些解能互达（分支结构）

判定用的是**两档证书**：`sign det J` 是严格的**单向**判据——异号必属不同分支；再用**保间隙的"见证走法"**做构造性证明，说明同号的两个解确实可以连通。工程上常用的**肩 / 肘 / 腕标签实测双向失效**：标签不同也能连通，标签相同也可能不连通。

落到工程上：同一条规定的笛卡尔闭环，十个起始姿势各自能跟踪 **100% / 23.2% / 30.8% / 0.9%**，弧的端点正是已定位的 fold 穿越点。换句话说，可行路径是**唯一性域**的性质，而不是 aspect 的性质。

### 问题二：为什么没有闭式解、绕过它要付出什么代价

障碍是**可测的 0.136 m 腕部偏置**——最后三根轴不共点，经典闭式解的前提不成立。把该偏置清零，得到一台"几乎一样"的可解邻机 **SR0**，再用**参数延拓**把解带回真机；这一步顺带量出了 fold 谱、实事件的 **±2 定律**和分裂指数 **0.39–0.48**，并实测 **16 解上界真的被达到**。代价写在适用边界上：**20–30%** 的位姿在 SR0 上根本没有种子（另有一个"三连续轴平行"的可解邻机，能覆盖这些盲位姿）。

### 问题三：能不能用根号写出来

**不能。** 让参数绕折点走一圈，把解与解之间的交换拼起来，得到的就是**单值化群**。SR5 的复单值化群实测含 **`S6`**（11 条对换边 ⇒ 连通分量 `[6,6,1,1,1,1]`，**两个 6 元分量各自闭包到阶 720 = |S6|**，构成双份独立证据）⇒ 群不可解 ⇒ 逆解无根式解。两条限定：这不排除闭式消元；且这是**复**侧的结论，与实侧分支结构**正交**。

同一套仪器在 **SR0** 上测到第一个非恒等对换（四对换的对合，八条根身份全程存活），测得子群**阶 2、阿贝尔 ⇒ 可解**。判据正好把两台臂分在几何预言的两侧。

### 入口与配套

* [`study/summary/README.md`](study/summary/README.md) —— **零上下文导读**：五篇文档（总览 + 唯一性域 / A→B + 可解性与 cusp + 单值化群 + 仪器与未决），配 **5 张由实测数据生成的图**（脚本在 [`study/doc_figures.py`](study/doc_figures.py)，用 `python3 -m study.doc_figures` 重画），另含术语表、结论账本与最少复现命令集。
* [`study/paper.pdf`](study/paper.pdf) —— 学术版（7 页）。
* [`study/NOTES.md`](study/NOTES.md) —— 逐日详细日志，含被推翻的结论与 17 条仪器失误档案。
* [`study/REPORT.md`](study/REPORT.md) —— 十分钟中文简报。
* [`study/collab/QUESTIONS.md`](study/collab/QUESTIONS.md) —— 编号问题清单（状态 / 命令 / 数字）；[`study/collab/closing-exchange.md`](study/collab/closing-exchange.md) —— 收尾对谈。

研究用的仪器（沿任务路径提升整条纤维、保间隙见证走法、SR0 闭式求解器、3R 基准）都在 [`study/`](study/) 里，可以单独取用。

### 研究约定

[`study/`](study/) **只 import 库、绝不修改库**；每个实验都是可运行的模块（`python3 -m study.<module>`，位于 pytest 的 `testpaths` 之外），新实验编号先在 [`study/collab/QUESTIONS.md`](study/collab/QUESTIONS.md) 注册，每个论断都附上产生它的命令与实测数字，**失败与撤回照记不删**。

## 功能特性

- **DH 参数** —— 改进型（MDH）与标准型（SDH）两种约定，支持单位换算与逐机器人 YAML 描述。
- **正运动学** —— 由关节位置求末端位姿，两套后端都支持。
- **Jacobian 与 Hessian** —— CasADi 后端解析求导；NumPy 实现则在测试套件里与数值导数对照校验。
- **逆运动学** —— 四种求解器：阻尼最小二乘、冗余臂零空间求解器、单步伺服求解器，以及实验性的 QP 求解器。
- **可操作度** —— Yoshikawa 指标及其梯度，供冗余子任务使用。
- **两套后端** —— `RobotModelNumpy` 每次调用都现场计算稠密数组；`RobotModelCasadi` 把运动学、Jacobian 与各阶导数一次性编译好。
- **几何工具** —— 旋转矩阵、RPY 角、四元数与齐次变换。

## 环境要求

- Python >= 3.9
- [NumPy](https://numpy.org/) >= 1.21
- [CasADi](https://web.casadi.org/) >= 3.6
- [PyYAML](https://pyyaml.org/) >= 6.0

## 安装

```bash
git clone https://gitee.com/yangliangzhu_rob/py_robot_library.git
cd py_robot_library

# 可编辑安装走 PEP 660，需要 pip >= 23。
python -m pip install --upgrade pip
pip install -e .
```

开发时把测试与代码检查工具一并装上：

```bash
pip install -e ".[dev]"
```

## 快速开始

```python
import numpy as np

from model import IkType, ModelFactory

# 在 NumPy 后端上构建一台 7 自由度珞石 ER3。
robot = ModelFactory.create("er3", backend="numpy", ik_type=IkType.IK_STANDARD)
print(robot.num_dof)  # 7

# 正运动学与几何 Jacobian。
q = np.zeros(robot.num_dof)
pose = robot.fk(q)
jacobian = robot.jacobian(q)

# 逆运动学：给定目标位姿，从种子构型出发求解。
target = robot.fk(np.radians([10, 20, -15, 30, 5, -25, 40]))
solution, success = robot.ik(q, target)
assert success
print(np.allclose(robot.fk(solution), target, atol=1e-4))
```

## 使用说明

### 构建模型

`ModelFactory.create(name, backend=..., ik_type=...)` 会在 `model/configs/` 里查找机器人描述，返回一个配置好的模型。

| 名称 | 机器人 | 自由度 | 参数形式 |
|------|-------|-----|------------|
| `sr5` | ROKAE xMate SR5 | 6 | matrix |
| `sr5_v2` | ROKAE xMate SR5，限位收紧 | 6 | matrix |
| `er3` | ROKAE xMate ER3 | 7 | MDH |
| `er3_v2` | ROKAE xMate ER3，展开的 DH | 7 | MDH |
| `er3_sdh` | ROKAE xMate ER3 | 7 | SDH |
| `er3_plus` | ROKAE xMate ER3 Plus | 7 | MDH |
| `nerv_er3` | nerv ER3 | 7 | MDH |
| `aubo_c5` | AUBO C5 | 6 | MDH |
| `franka` | Franka Emika Panda | 7 | MDH |

两套后端都可用：

```python
from model import ModelFactory

numpy_robot = ModelFactory.create("er3", backend="numpy")
casadi_robot = ModelFactory.create("er3", backend="casadi")
```

两者给出的运动学在浮点精度内一致；CasADi 后端另外提供精确导数，编译表达式构建完成后速度更快。

### 手写机器人描述

```python
import numpy as np

from model import Dh, IkType, RobotModelNumpy, get_matrix_list

dh = Dh(
    [
        # [d, alpha, a, theta]
        [0.150, 0.0, 0.0, 0.0],
        [0.0, -np.pi / 2, 0.0, 0.0],
        [0.0, np.pi / 2, 0.0, 0.0],
        [0.610, -np.pi / 2, 0.0, 0.0],
        [0.0, np.pi / 2, 0.0, 0.0],
        [0.110, -np.pi / 2, 0.0, 0.0],
    ],
    dh_type="mdh",
)

config = {
    "param": dh,
    "lower": np.radians([-170, -120, -170, -170, -120, -170]),
    "upper": np.radians([170, 120, 170, 170, 120, 170]),
}

robot = RobotModelNumpy()
robot.build("dh", IkType.IK_STANDARD, config)
```

一台机器人由一条有序变换列表 `Ms = [base, link_1, ..., link_n, tool]` 描述。`get_matrix_list` 从 DH 表构造它；`"mat"` 布局则直接传入各连杆变换。

### 逆运动学

| `IkType` | 行为 |
|----------|-----------|
| `IK_STANDARD` | 带自适应步长回退的阻尼最小二乘；约束完整位姿。 |
| `IK_NORMAL` | 同上，但放松工具偏航角，有助于 7 自由度臂收敛。 |
| `IK_NULL` | 约束完整位姿，并在 Jacobian 零空间里加一个冗余子任务：先避开关节限位，再最大化可操作度。 |
| `IK_NULL_NORMAL` | `IK_NULL` 的放松偏航角版本。 |
| `IK_NAIVE` | 只做一步阻尼最小二乘；用在伺服回路里。 |
| `IK_QP` | 实验性的单步 QP 形式。 |

所有求解器调用签名相同，返回 `(joint_positions, success)`：

```python
from model import IkType, ModelFactory

robot = ModelFactory.create("er3", backend="casadi", ik_type=IkType.IK_NULL)
solution, success = robot.ik(seed, target_pose)
```

逆解是迭代且依赖种子的：请传入接近预期解的构型。`IK_NORMAL` 与 `IK_NULL_NORMAL` 两个变体只在位置上收敛，按设计不匹配目标偏航角。

### 工具变换

`set_tool` 在法兰上安装可互换工具。它替换上一件工具，而不是在其上叠加。

```python
tool = np.eye(4)
tool[2, 3] = 0.1  # 沿工具 Z 轴 10 cm
robot.set_tool(tool)
robot.set_tool(np.eye(4))  # 再把它卸掉
```

### 几何工具

```python
import numpy as np

from tools.geometry import nervCartToAffine, quat2rot, rot2quat, rot_x, rpy2rot

rot_x(np.pi / 2)                      # 绕 X 轴的 3x3 旋转
rpy2rot([0.1, 0.2, 0.3])              # ZYX 复合，弧度
quat2rot([1, 0, 0, 0])                # [w, x, y, z] -> 3x3
rot2quat(np.eye(3))                   # 3x3 -> [w, x, y, z]

# nerv 笛卡尔位姿为 [x, y, z, rx, ry, rz]，单位是毫米与度。
nervCartToAffine([100.0, 200.0, 300.0, 0.0, 0.0, 90.0])
```

## 测试

测试套件使用 [pytest](https://pytest.org/)：

```bash
pip install -e ".[dev]"

pytest                      # 全量套件
pytest tests/test_ik.py     # 单个文件
pytest -k manip             # 按名字筛选
```

模型层测试在两套后端上都做了参数化，因此 NumPy 与 CasADi 实现一旦出现分歧，构建就会失败。Jacobian 与可操作度梯度都是与数值导数对照校验的，而不是与存档的参考值比较。

各机器人模型的几何回归数据放在 `test_data/*.npz`，由 `tools/generate_test_data.py` 重新生成。

## 代码检查

```bash
ruff check .        # 检查
ruff check --fix .  # 应用安全修复
```

## 项目结构

```
py_robot_library/
├── model/                  # 包：机器人模型与运动学
│   ├── __init__.py         # 公开 API
│   ├── dh_param.py         # DH 表与连杆变换
│   ├── ik_type.py          # IkType 枚举
│   ├── ik_solver.py        # IK 求解器 + 求解器工厂
│   ├── ik_srs.py           # S-R-S 型 7 自由度臂的解析逆解
│   ├── mrobotics.py        # 旋量法刚体辅助函数
│   ├── robot_model_base.py # 共用的模型配置逻辑
│   ├── robot_model_numpy.py
│   ├── robot_model_casadi.py
│   ├── model_factory.py
│   ├── common.py           # 向后兼容的聚合出口
│   └── configs/            # 逐机器人 YAML 描述
├── tools/                  # 几何辅助函数与独立脚本
├── study/                  # 逆解分支结构研究（见上文对应章节）
│   ├── summary/            #   零上下文导读：5 篇文档 + 5 张实测图
│   ├── doc_figures.py      #   用实测数据重画 summary 的 5 张图
│   ├── taskspace.py        #   沿任务路径提升整条纤维（诞生、fold、护栏）
│   ├── chamber.py          #   关节空间里的保间隙见证走法
│   ├── sr0.py              #   可解邻机的闭式求解器
│   ├── cuspidal3r.py       #   3R 基准臂（cuspidal，且有 cusp）
│   ├── exp*.py             #   各号实验，编号登记在 collab/QUESTIONS.md
│   ├── NOTES.md            #   详细日志，含撤回
│   ├── REPORT.md           #   十分钟中文简报
│   ├── paper.typ/.pdf      #   学术版
│   └── collab/             #   双人协作记录：问题清单、轮次日志、往来信件
├── tests/                  # pytest 套件
└── test_data/              # 几何回归数据
```

## 约定

- 除函数另有说明外，角度一律用弧度。
- 旋转矩阵为 3x3；齐次变换为 4x4。
- 四元数为 `[w, x, y, z]`。
- RPY 角为 `[roll, pitch, yaw]`，对应 ZYX 复合
  `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`。
- 关节限位一律以弧度存储，不论配置文件里声明的是什么单位。
- `study/` 属于探索区：可以 import 库，但绝不修改库。实验都是 `python3 -m study.<module>` 形式的模块，
  新实验编号先到 `study/collab/QUESTIONS.md` 注册，结论写在 `study/NOTES.md` / `study/REPORT.md`。
  失败与撤回照记不删。

## 贡献

欢迎贡献。提合并请求之前，请先读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

[MIT](LICENSE)
