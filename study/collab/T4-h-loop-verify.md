# T4 h 环验证 — exp68：半径 0.015 扫描结果 h 的完整 Gate A 控制

执行者：`h-loop-verify`（T4 分工，2026-10-17）。范围：**只认证 h 的测量协议**；非可解判定（Gate B
群论）与路径认证（Gate C）不在本任务。验收以 `GPT_GATE_B_WITNESS_20261017.md` §3 的「下一棒任务」为准：
正环必须复现 h、逆环必须读 **h⁻¹ 而非 h**（h 含 3-循环、非对合）、空环恒等、96→192 加密一致，
冻结抛光根表与 exp66 全程纪律，逐步原始数据归档，任何不一致 exit 非零。

## 1. 结论（先读这个）

h = **(1 8)(4 10 13)(7 12)(9 15)**，定位自 exp66 归档 **fold1**（−x 切片，
`u* = +0.22628458830430512`（实轴），半径 0.015 扫描，`fold1_scan_r0.015`）。本次把它在完整
Gate A 控制协议下重放：同一冻结抛光根表（直接从归档装载、编号不变、重抛光移动 0.0e+00）、
同一条 669 节点路径（由 `build_loop(0, u*, 0.015, samples=96, orientation=+1, arc=0.05)` 重推导，
与归档逐位相等后断言通过）、exp66 纪律（切线预测、跳盆拒收 0.5 rad、近碰撞地板 1e-2、终点单射
匹配、未匹配=失败态）。

**9/9 自检 PASS、exit 0、四环 0 failures。四条环的完整置换：**

| 环 | 置换（image notation） | 循环型 |
|---|---|---|
| main（96 采样，669 节点） | `[0,8,2,3,10,5,6,12,1,15,13,11,7,4,14,9]` | **h** = (1 8)(4 10 13)(7 12)(9 15) |
| reverse | `[0,8,2,3,13,5,6,12,1,15,4,11,7,10,14,9]` | **h⁻¹** = (1 8)(4 13 10)(7 12)(9 15)，**不是 h** |
| empty（r=1e-4，101 节点） | `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]` | 恒等 |
| refined（192 采样，1341 节点） | `[0,8,2,3,10,5,6,12,1,15,13,11,7,4,14,9]` | **h**（与 main 一致） |

reverse = main⁻¹、empty = 恒等、refined = main、main∘reverse = 恒等、main = 归档 h，全部 PASS；
每条环 16/16 根匹配且单射、0 failures。**h 现在与 a=(1 6)、d=(0 14)(4 6) 同级**：三条基础回路
的数值单值化测量齐备，GPT 条件见证（纯共轭得 (1 6),(6 8),(1 4),(8 10) → S5，阶 120）的前提
全部满足。非可解结论的判定仍属 Gate B，路径认证仍属 Gate C。

## 2. fold 与路径的定位（重放来源，不重新精化）

- **fold**：`folds_json` 条目 1 — 方向 −x，播种对标签 (2,6)（event `u=+0.22639` jump、检测间隙
  0.307；T4 §6 已记此标签很可能是贪心伪配对，`u*` 处的真实 fold 对是 (1,8)），精化
  `u* = +0.22628458830430512 + 0i`，增广残差 **5.545e-11**，实轴。
- **路径构造**：`build_loop(0, u*, 0.015, samples=96, orientation=+1, arc=0.05)`——上半平面弧
  `Im=0.05` 进圈（避开复根的实轴 birth fold）、圆从实侧 `u*−0.015=+0.211285` 起绕一整圈、原路
  返回；669 节点，与归档 `fold1_scan_r0.015_u_real/u_imag` **逐位相等**（`np.array_equal` 断言，
  重放前强制核对，任何路径构造漂移都会 exit 非零而非静默换路）。
- **半径阶梯**（归档 `fold1_scan_r*`，本任务不重测）：`1e-4 → (1 8)(9 15)`；
  `1e-3/5e-3 → (1 8)(7 12)(9 15)(10 13)`；`1.5e-2 → h`；`3e-2 → (1 8)(2 15 9)(4 10 13)(7 12)`。
  ⇒ r=0.015 圆内是**一个分支点簇**（(1,8)@u* 及 (7,12)/(9,15)/3-循环因子），h 是按一个乘积置换
  整体记录、整体使用，**未拆分因子**。

## 3. 一致性数字（四条环）

| 环 | 节点 | 失败 | max 逐根残差 | max 修正 | max 实际步 | 沿程 min 根间隙 | 终点 max 残差 | 终点到像 max 间隙 |
|---|---|---|---|---|---|---|---|---|
| main | 669 | 0 | 9.99e-10 | 5.4e-2 | 6.1e-2 | 0.458212168 | 9.4e-10 | 3.7e-9 |
| reverse | 669 | 0 | 9.99e-10 | 9.3e-2 | 6.1e-2 | 0.458212168 | 9.4e-10 | 3.7e-9 |
| empty | 101 | 0 | 2.5e-10 | 3.3e-3 | 3.2e-3 | 0.450715474 | 5.3e-11 | 4.7e-10 |
| refined | 1341 | 0 | 1.0e-9 | 2.7e-2 | 3.0e-2 | 0.458212168 | 9.1e-11 | 2.3e-10 |

冻结基点：重抛光最大移动 **0.0e+00 rad**、逐根残差 ≤ **1.6e-14**、最小间隙
**0.458212168 rad（对 (14,15)）**，与归档 `base_min_gap`/`base_closest_pair` 一致；基点
`p0 = [0.082144, 0.101203, 0.784582] m`（seed 0 精确复现）、`links[4][1,3] = 0.136 m` 全程不动。
main/reverse 沿程最小根间隙 = 基点最小间隙（环上从未比基点更接近合并）。

9 项自检（`checks_json` 原文）：main/reverse/empty/refined 各完整置换（4 项）、
`main reproduces the archived h`、`reverse == main^-1`、`reverse is h^-1, not h (h has a 3-cycle)`、
`empty loop is identity`、`step-refinement agrees with main`、`main . reverse == identity` —— 全 PASS。

**失败护栏实测（负对照）**：`--empty-radius 5e-3` 重跑 → 空环读
`[0,1,2,3,4,-1,6,7,8,-1,10,11,12,13,15,14]`（根 5、9 在 node 2 跳、残留 (14 15) 读数，复现 exp66
归档 `emptyprobe_-x_r0.005`），`empty loop is a complete permutation` FAIL、**exit 1**。证明 1e-4
半径是必需的、且不一致不会被静默放过。

## 4. 每步原始数据清单（`study/exp68_h_loop.npz`，5.5 MB，98 数组）

复数值一律实部/虚部分开存、不截断。**每条环（main/reverse/empty/refined）的每个参数节点**都存：

| 键（`{tag} ∈ main,reverse,empty,refined`） | 形状 | 含义 |
|---|---|---|
| `{tag}_u_real/u_imag` | (N,) | 参数路径实/虚部（N=669/669/101/1341） |
| `{tag}_roots_real/roots_imag` | (N,16,6) | 16 根复轨迹实/虚部（每个参数节点） |
| `{tag}_residual` / `_correction` / `_actual_step` | (N,16) | 逐根残差 / 修正 / 实际步长 |
| `{tag}_min_gap` / `_closest_pair` | (N,) / (N,2) | 每节点最近对间隙与配对 |
| `{tag}_alive` / `_merge_risk` | (N,16) / (N,) | 存活标志 / 近碰撞标志（本次全 0） |
| `{tag}_permutation` / `_complete` | (16,) / () | 置换与单射完备性 |
| `{tag}_failures_json` | () | 失败记录 JSON（本次全 `[]`，原样保留） |
| `{tag}_q_end_real/imag`、`_endpoint_residual`、`_gap_to_image` | (16,6)/(16,) | 终点构型、终点逐根残差、到像间隙 |

另有：`target0`/`direction`/`u_star`/`radius`/`arc`/`samples`/`refined_samples`/`empty_radius`/
`jump_tolerance`/`merge_floor`（协议参数）、`fold_pair`/`fold_event_u`/`fold_event_gap`/
`fold_event_kind_json`/`fold_refine_residual`（fold 出处）、`fibre_real/imag`/`fibre_residual`/
`fibre_polish_move`/`base_min_gap`/`base_closest_pair`（冻结抛光根表）、`expected_h`/`expected_h_inv`
（对照期望）、`checks_json`/`all_ok`/`summary_json`（`machine_verified=False`）。

## 5. 未决（诚实记录，不默认成固定点）

1. **空环半径必须是 1e-4**：exp66 空环探针实测 −x 切片有离轴 (14,15) 分支点在距基点
   **2e-3..3.2e-3** 内（`emptyprobe_-x_r0.005`：根 5、9 跳、含 (14 15) 读数、complete=False；
   r≤2e-3 恒等）。故本协议空环固定 **1e-4**（低于所有实测近基分支点）；本任务负对照（§3）已
   实测 r=5e-3 空环会以失败态暴露。**任何更大半径的空环读数都会是 branch point 证据而非缺陷。**
2. **h 非对合**：逆环必须用真实逆 **h⁻¹ = (1 8)(4 13 10)(7 12)(9 15)**（本协议如此且 PASS）；
   GPT 文档已强调，共轭计算不得把 h⁻¹ 写成 h。
3. **播种对标签 (2,6) 是伪配对**（T4 §6 已记；精化落在 (1,8) 簇）。测量以实测置换为准，
   标签仅作出处保留在 `fold_pair`。
4. **簇成员未逐一精化**：r=0.015 圆内的 (7,12)/(9,15) 与 3-循环因子只由半径阶梯定位；
   h 按乘积置换整体记录（审计允许共享临界值按一个乘积置换记录，已遵守）。
5. **全部为数值单值化测量**（`machine_verified=False`）；Gate C 路径认证未做。Gate B 条件见证
   （`exp67_group_witness`：`hah⁻¹=(6 8)` 等四纯对换、显式闭包 120）的前提「h 有完整控制」
   现已满足，但非可解结论的最终判定仍属 Gate B 群论。

## 6. 边界（与任务书一致）

本交付只认证 **h 的测量协议**（正/逆/空/加密四环 + 逐步原始数据）。非可解子群见证是 Gate B
的目标证据、路径认证是 Gate C —— 均不在本任务范围。

## 7. 命令与复核

```bash
cd py_robot_library
python3 -m study.exp68_h_loop    # exit 0；9/9 自检 PASS、0 failures（~2 min）
python3 -m study.exp68_h_loop --empty-radius 5e-3   # 负对照：exit 1（空环被近基 (14,15) 分支点污染）
python3 -m study.exp67_group_witness   # 仍 exit 0（归档未动，条件见证的前提由本任务补齐）
python3 -m ruff check .          # All checks passed
python3 -m pytest -q             # 217 passed
```
