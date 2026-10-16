# T1 结果：复延拓跟踪器的终点契约（复现 / 修复 / 爆炸半径审计）

> 日期：2026-10-16。执行者：DeepSeek Harness（teammate `tracker-audit`）。任务书：[T1](GPT_EXECUTION_TASKS_20261016.md)，
> 依据：[数学复核 §3](GPT_MATH_REVIEW_20261016.md)。分支 `study/ik-branch-and-continuation`，**未提交**，改动留在工作区。
> 环境：Python 3.10.12 / numpy 2.2.6 / 24 核；命令均在仓库根目录执行（注明"基线副本"者除外）。
> 计时受并行任务数（同时 6–11 个实验进程）影响，只用于量级参考。

## 0. 结论摘要

1. **缺陷已复现并已修**：`exp38.track_path` 与 `exp37.track_circle` 的"成功"过去只表示"有一个子步被接受"，
   不表示走到了请求终点。修复后 `ok=True` 蕴含（并复核）终点到达；缩短后的步只是**子步**，
   会一直走到该段终点，或撞到步长/调用预算地板后返回**实际到达的参数**与 `ok=False`。
2. **探针**：GPT 的原命令修复前输出 `[0.0625+0.j] 0.0 True`，修复后 `[1.+0.j] 0.0 True`；
   回归套件 `study/exp60_tracker_contract.py` 从 **3/11 通过** 变为 **11/11 通过**。
3. **重跑对照**（HEAD 修复前 vs 修复后，同一命令、同一随机种子）：
   * **逐字不变**：`exp38`、`exp40`（400 种子）、`exp44`、`exp46`；
   * **不变但精度改善**：`exp37` 正结果（对换 `[1,0]` 两旋向都给；距离 1e-10 → 1e-14，来自输入根抛光）；
   * **变化**：`exp45`（伪边 `(15,14)` 消失；fold 0.655086 由 MISMATCH 变 MATCH；另三个 fold 仍读不出）；
   * **变为有效**：`exp37` 负对照（修复前是空洞的，见 §4.2）。
4. **调用者审计**：唯一**独立复核终点**的调用点是 `exp38` phase A.3（在 s=1 重算 12 分量残差并卡 1e-8）——
   它在旧契约下也是安全的。其余 9 个调用点只信 `ok` + 最近根匹配（阈值 1e-2），**有风险**；
   其中 `exp45` 的读数确实被旧契约污染（§4.3）。
5. **历史记录**：`exp37` 的"空圆负对照不动 / 双对照干净"这条证据**已证伪**（当时 3/4 次调用根本没动），
   已用新基点的真实纤维重做并通过（12/12）。fold 边表、exp40 分支点**数目**、exp45 复核三条边记 **有风险**；
   两处差异**未决**并给出可结算命令（§6）。

## 1. 复现（修复前）

### 1.1 GPT 的原探针（逐字）

```bash
python3 - <<'PY'
from unittest.mock import patch
import numpy as np
from study.exp38_monodromy_group import track_path

def fake_links(base, offset, z):
    return z

def fake_newton(z, q, target):
    return np.array([z], dtype=complex), 0.0

with patch('study.exp38_monodromy_group.links_at', fake_links), \
     patch('study.exp38_monodromy_group.newton_complex', fake_newton):
    q, residual, ok = track_path([], 1., np.eye(4), np.array([0j]),
                                 [0j, 1+0j], max_step=.1)
print(q, residual, ok)
PY
```

| | 输出 | 退出码 |
|---|---|---|
| 修复前 | `[0.0625+0.j] 0.0 True` | 0 |
| 修复后 | `[1.+0.j] 0.0 True` | 0 |

### 1.2 回归套件 `study/exp60_tracker_contract.py`（先登记编号，见 `QUESTIONS.md`）

探针原理：`links_at` 返回参数本身、`newton_complex` 把它当构型返回 ⇒ **跟踪器返回的值就是它停下的参数**，
残差恒为零，不可能"看起来接近"。每次调用同时跑**冻结的修复前控制流**（本模块内的 `legacy_*`，函数体与
基线副本逐字相同，已用 AST 比对）与**当前实现**，两列并排打印。圆的探针额外检查**未展开扫过角**：
`exp(iθ)` 回到基点不足以证明走完一圈。

修复前（`python3 -m study.exp60_tracker_contract`，exit 1）：

```
track_path -- the parameter actually reached (probe root q(s) = s)
  case                                                     legacy control |             current module  check
  [1] exact root, targets [0, 1]                           0.0625 ok=True |             0.0625 ok=True  FAIL
  [2] cornered polyline (no chord)                 0.1724+0.0898j ok=True |     0.1724+0.0898j ok=True  FAIL
  [3] the same path walked backwards               0.8276+0.4102j ok=True |     0.8276+0.4102j ok=True  FAIL
  [4] last segment needs subdivision                       0.1094 ok=True |             0.1094 ok=True  FAIL
  [5] corrector dies past s = 0.5                          0.0625 ok=True |             0.0625 ok=True  FAIL
  [6] corrector never converges                           0.0000 ok=False |            0.0000 ok=False  PASS
  [7] single-point path is a no-op                         0.7000 ok=True |             0.7000 ok=True  PASS

track_circle -- the unwrapped angle actually swept (probe root q(s) = s)
  [8] full turn, orientation +1, 96 samples                 swept +0.1560 |              swept +0.1560  FAIL
  [9] full turn, orientation -1, 96 samples                 swept -0.1560 |              swept -0.1560  FAIL
  [10] full turn from only 8 samples                        swept +0.0061 |              swept +0.0061  FAIL
  [11] corrector dies in the lower half-plane               swept +0.1560 |              swept +0.1560  PASS

3/11 contract checks pass
```

修复后（exit 0）：

```
  [1] exact root, targets [0, 1]                           0.0625 ok=True |             1.0000 ok=True  PASS
  [2] cornered polyline (no chord)                 0.1724+0.0898j ok=True |     1.0000+0.5000j ok=True  PASS
  [3] the same path walked backwards               0.8276+0.4102j ok=True |             0.0000 ok=True  PASS
  [4] last segment needs subdivision                       0.1094 ok=True |             1.0000 ok=True  PASS
  [5] corrector dies past s = 0.5                          0.0625 ok=True |             0.5000 ok=False  PASS
  [6] corrector never converges                           0.0000 ok=False |            0.0000 ok=False  PASS
  [7] single-point path is a no-op                         0.7000 ok=True |             0.7000 ok=True  PASS
  [8] full turn, orientation +1, 96 samples                 swept +0.1560 |              swept +6.2832  PASS
  [9] full turn, orientation -1, 96 samples                 swept -0.1560 |              swept -6.2832  PASS
  [10] full turn from only 8 samples                        swept +0.0061 |              swept +6.2832  PASS
  [11] corrector dies in the lower half-plane               swept +0.1560 |              swept +3.1416  PASS

11/11 contract checks pass
```

要点：`[3]` 反向路径修复前停在 `0.8276+0.4102j` 且 `ok=True`；`[5]` 校正器在 0.5 之后永不收敛，
修复前报 `ok=True, q=0.0625`（既没到 0.5 也没到 1），修复后 `ok=False, q=0.5000`（正是"最后真正到达的参数"）；
`[8]`–`[10]` 说明 `max_step` 小于单步所需时，旧代码只扫过 1/40 圈（96 采样）或 1/1000 圈（8 采样）却报成功。

## 2. 修复（改动最小、与周围风格一致）

### 2.1 `study/exp38_monodromy_group.py::track_path`

* 新增仅关键字参数：`max_subdivisions=64`（单步最多折半次数）、`min_span=1e-9`（步长地板）、
  `max_calls=20000`（整条路径的校正器调用预算）。原签名与三个返回值不变。
* 内层由 `for _ in range(8): ... break` 改为：**每段用 `while cursor != goal` 迭代到终点**；
  拒绝时只把当前 `span` 折半（`trial = goal - (remaining - span)`，当 `span == remaining` 时
  `trial` 恰为 `goal`，所以整段被接受时**逐位落在终点**，不会用弦跨过折角）；
  地板/预算耗尽 ⇒ `return q, worst, False`（`q` 是最后接受的点）。
* **终点残差复核**：走完后在最后一个请求参数上再做一次校正器调用，要求
  `residual < tolerance` 且 `norm(candidate - q) <= max_step`（后者防终点跳盆），否则 `ok=False`。
* docstring 改成新契约（含 Args/Returns）。

### 2.2 `study/exp37_complex_loop.py::track_circle`

* 同样三件套（子步迭代到该采样角、`max_subdivisions`/`min_span`/`max_calls` 预算、终点复核），
  并**按未展开角**推进（不再每步从 `angle_prev` 起算的隐式漂移）；最后一个检查点就是
  `start_angle + orientation * 2π`。
* 第三个返回值语义改为**未走完的角（弧度）**：`0.0` 才算走完整圈；撞地板/预算返回尚欠角度。
  旧的第三个值是"到起点根的距离"，部分回路也有限，调用者只测 `isfinite` ⇒ 无法区分。
  调用者 `exp37` 已同步改为 `unfinished > 0.0` 判失败（全仓库只有它调用）。

### 2.3 `study/exp37_complex_loop.py` 负对照（GPT 指出的第二个缺陷）

* 新基点取纤维：`centre = 0.85`，基点 `s0 = 0.85 + 1e-3`（角 0），在 `sr0.robot(path(s0))` 上普查、
  用**同一复校正器抛光**每个候选根到 12 分量残差 `<1e-8`（LM 普查只到 ~1e-5，
  抛光前后实测 9.7e-07 / 1.2e-05 → 6.4e-16 / 7.9e-16）、按构型距离去重，再绕圈与**同一基点**的纤维比较。
* 主回路的输入根（2 个 pair + 2 个 control）同样先抛光并验证满足基点方程：
  抛光前 5.3e-11 / 1.4e-11 / 9.7e-07 / 1.2e-05 → 抛光后 ≤2.8e-15，最大移动 8.9e-06；
  若不满足（残差 >1e-8 或移动 >0.1）直接中止。这条检查在第一次修复运行时就命中
  （见 §4.2：原先直接用 LM 根会以 1.2e-05 中止，说明"输入根确实满足起点方程"这条验收项有必要）。

## 3. 调用者审计（`grep -rn "track_path\|track_circle" study/`）

同型"缩短后 break"的实现检查（T1 步骤 5）也一并做了：`homotopy.track`、`exp21_endgame`、`exp35_cusp`、
`exp55_3r_cusp_check`、`exp58_sr0_fold_and_group`、`exp59_sr0_radial_fold`、`chamber.clearance_path`、
`exp18`、`exp23` 在拒绝后都是**重试同一步**（`continue`）或返回失败，**没有第二类命中**——
缺陷只在上面两个"外层遍历目标点、内层 break 出去"的函数里。

| 调用者（模块 :: 函数 : 行） | 独立复核终点？ | 可能被早退污染的记录结论 | 证据（命令 + 数字） |
|---|---|---|---|
| `exp38 :: main` phase A.3 :225 | **是**：在 s=1 重算 `residual12(...)` 并卡 `<1e-8`，再要求 `ok` | 16 根纤维里的 6 个复根 | 滞后端点在 s=1 的残差远大于 1e-8（探针 [1] 的端点停在 0.0625）；该分支前后逐字相同：残差 1.3e-14/1.6e-14/6.1e-15，`|imag|` 0.60/0.91/0.15，纤维 16 |
| `exp38 :: main` phase B :243 | 否：只用 `ok` + 环面距离 `<1e-2` | **Q7/Q15 的 fold→对换边表与分量结构** | 前后逐字相同（8 边、`[4,3,3,2,1,1,1,1]`）；插桩审计：该轮 16 次调用 **0 次折半**、3316 个请求点全命中（§4.4）⇒ 这一轮旧契约未触发；但见 §4.5：记录表的两条配对与此不同 |
| `exp40 :: main` phase 0 纤维 :130 | 否（同 phase B 的匹配） | exp40 自建的 16 根纤维 | 前后 16 根不变（同一 s=1 构造） |
| `exp40 :: main` phase 1 种子 :146 | **部分**：`refine_branch` 自带 1e-9 残差门、且 s 是自由未知量，但**种子**来自 tracker | 复分支点**集合**（记录 14 / 6 的争议） | 400 种子前后同一集合（2 点：1.070113±0.013841i，残差 1.3e-10/1.9e-10）；converged 20 vs 19 of 398 |
| `exp40 :: main` phase 2 回路 :185 | 否：`ok` + 最近根 `<1e-2` | 复分支点边 `(5,8)/(8,5)` | 前后逐字相同（moved `[(5,8),(8,5)]`，untracked 0） |
| `exp45 :: main` 纤维 :79 | 否 | exp45 自建纤维 | 前后 16 根不变 |
| `exp45 :: main` 回路 :110 | 否：`ok` + 最近根 `<1e-2` | **exp45 的四条对抗性复核边** | **前后不同**，见 §4.3；插桩审计：64 次调用 **5522 次折半**、漏走 **27263** 个请求点却 44 次 `ok=True`（§4.4）⇒ 旧读数确实被污染 |
| `exp46 :: main` 纤维 :117 / stratum 端点 :136 | 否 | kimi 的 `(5,9)` 边（fold 1.047164） | 前后逐字相同：1 个 stratum @1.047164（残差 4.0e-12），endpoints `[5, 9]` |
| `exp37 :: main` 主回路 :301（`track_circle`） | 否（修复后靠 `unfinished`，仍无残差复核） | **Q15 的对换 `[1,0]`** | 仪器审计（§4.2）：修复前该调用 8/8 次都是 96 子步、**0 次拒绝**、扫过 ±6.2832 rad ⇒ 缺陷未触发；修复后同结果 |
| `exp37 :: main` 负对照 :357（`track_circle`） | 否 | **"空圆负对照不动"这条记录** | **已证伪**（§4.2）：修复前 3/4 次调用 0 子步、8 次拒绝、`|dq|=0`（原样返回 q0），第 4 次拿 fold 的纤维比对 |

## 4. 重跑对照（HEAD 修复前 vs 修复后）

"基线副本"= 修复前的仓库快照（HEAD 就是修复前的提交，改动未提交：`git archive HEAD | tar -x -C <dir>`），
命令与正式仓库完全一致。本次运行的原始日志与 JSON 已归档到
`/home/yang/workspace/research/.t1_evidence/`：修复前 `*_before.log`、修复后 `*_after.log`、
插桩审计 `*_audit*.log` + `*_lag*.json(.gz)`、day06 归档 `exp38_day06.log`。

### 4.1 逐字不变的三个半

| 实验 | 命令 | 修复前 | 修复后 |
|---|---|---|---|
| `exp38` | `python3 -m study.exp38_monodromy_group` | 8 folds；fibre 16；A.3 残差 1.3e-14/1.6e-14/6.1e-15；边 `(10,0)(7,12)(15,9)(8,1)(10,11)(12,13)(14,15)(9,5)`；分量 `[4,3,3,2,1,1,1,1]`；退出码 0，7m07s | **逐字相同**；退出码 0，8m03s |
| `exp46` | `python3 -m study.exp46_strata` | 1 stratum @1.047164（4.0e-12）；endpoints `[5,9]`；退出码 0，6m59s | **逐字相同**；退出码 0，7m00s |
| `exp40 --seeds 400` | `python3 -m study.exp40_complex_branches --seeds 400` | 2 复分支点（1.070113±0.013841i）；边 `[(5,8),(8,5)]`；converged 20/398 | 同集合、同边；converged **19**/398；退出码 0，8m25s |
| `exp44` | `python3 -m study.exp44_group_from_edges` | 10 边 ⇒ 分量 `[6,3,3,1,1,1,1]`，阶 25920，0.018s | **逐字相同**，0.014s |

注：`exp38`/`exp40`/`exp46` 的"逐字相同"是**修复前后**的比较（同一 HEAD 代码、同一 rng 种子）；
它说明这一次运行里旧契约没有改变可读数字，**不等于**历史记录表就是这个数字（见 §4.5）。

### 4.2 `exp37`：正结果不变、负对照从空洞变真实

命令 `python3 -m study.exp37_complex_loop`（fold 0.718354，radius 1e-3，samples 96）。

修复前（基线副本，exit 0，1m55s）：

```
  root 0 (pair): -> root 1 (distance 2.6e-10, |imag| 7.7e-18, worst residual 9.6e-15)
  root 1 (pair): -> root 0 (distance 9.0e-10, |imag| 8.7e-18, worst residual 9.2e-15)
  root 2 (control): -> root 2 (distance 4.4e-06, ...)
  root 3 (control): -> root 3 (distance 8.9e-06, ...)
  permutation [1, 0, 2, 3]: TRANSPOSITION          （两旋向相同）
negative control: same radius, centre away from every fold
  root 0: -> root 0 (distance 0.0e+00)
  root 1: -> root 1 (distance 0.0e+00)
  root 2: -> root 2 (distance 0.0e+00)
  root 3: -> root 3 (distance 6.7e-02)
```

**决定性仪器**：把修复前的 `track_circle` 包一层记录（`audit_circle.py`，只记录不改行为；同一次运行的实验输出与
基线逐字相同，证明透明）。下表的行号 220/248 是**修复前文件**的行号（修复后同一调用点在 301/357）：

```
caller                     centre steps refused     swept unfinished      |dq|    signal
exp37_complex_loop.py:220      0.718    96       0   +6.2832     0.0000   1.4e-01   1.4e-01
exp37_complex_loop.py:220      0.718    96       0   -6.2832     0.0000   1.4e-01   1.4e-01
   ...（8 次主回路调用全部 96 子步 / 0 拒绝 / 扫过 ±2π）
exp37_complex_loop.py:248      0.850     0       8   +0.0000     6.2832   0.0e+00       inf
exp37_complex_loop.py:248      0.850     0       8   +0.0000     6.2832   0.0e+00       inf
exp37_complex_loop.py:248      0.850     0       8   +0.0000     6.2832   0.0e+00       inf
exp37_complex_loop.py:248      0.850    96       0   +6.2832     0.0000   6.7e-02   6.7e-02
```

⇒ 两条结论：
* **主回路的对换 `[1,0]` 不受缺陷影响**：那时 8 次调用一次都没触发折半（96 子步、0 拒绝），走的是整圈；
* **负对照是空洞的**：4 次里 3 次 `accepted=0`、`refusals=8`、`|dq|=0` ⇒ 跟踪器压根没动，
  返回的 `q_end` 就是输入的 q0，于是打印的 `distance 0.0e+00` 被读成"自返"；
  第 4 次确实走了一圈（96 子步、扫过 +6.2832），但与 **fold 的纤维**比较，6.7e-02 无意义。

修复后（exit 0，2m40s）：输入根抛光（pair 5.3e-11→2.8e-15；control 9.7e-07→6.4e-16、1.2e-05→7.9e-16）；
主回路两旋向仍是 `[1, 0, 2, 3] = TRANSPOSITION`，距离 **1.3e-14 / 8.7e-15**（更干净，来自抛光而非契约）；
负对照在新基点自己的纤维上重建 **12 根**（s=0.851 处实纤维应是 12 = 10 + 2，两个消亡 fold 之后），
**12/12 全部自返**：最大距离 4.8e-13，`|imag|` ≤ 3.2e-16，`unfinished = 0.0`，最差残差 1.5e-15。

⇒ **负对照仍然通过**，而且这次是真的：12/12 根走完整圈回到自己。

### 4.3 `exp45`：读数改变（唯一被修复改变**读数**的实验；exp37 正结果不变、负对照由空洞变有效）

命令 `python3 -m study.exp45_fold_loops`（4 个 fold、全 16 根、`max_step=0.02`）。

| fold | 修复前 | 修复后 |
|---|---|---|
| 0.655086 | moved `[(0,10),(15,14)]`，untracked 8 ⇒ **MISMATCH** | moved `[(0,10)]`，untracked 10 ⇒ **MATCH** |
| 0.656878 | moved `[(7,12),(9,15),(15,14)]`，untracked 6 ⇒ MISMATCH | moved `[(7,12),(9,15)]`，untracked 8 ⇒ MISMATCH |
| 0.718354 | moved `[(15,14)]`，untracked 4 ⇒ MISMATCH | moved `[]`，untracked 6 ⇒ MISMATCH |
| 0.799244 | moved `[(15,14)]`，untracked 2 ⇒ MISMATCH | moved `[]`，untracked 4 ⇒ MISMATCH |

`(15,14)` 是 fold 0.961848 的边——修复前它在每个 fold 的读数里都出现，正是记录里说的"途中 fold 弧顶跳盆污染"；
修复后它消失，fold 0.655086 恢复出记录 R25 里那条干净边 `(0,10)`。**另三条边在本 HEAD 上读不出**
（untracked 8/6/4），这不是终点契约能解决的，属于已登记的 Q17（回路法需要预测-校正内核）。
⇒ exp45 的"独立复核"在修复后只能支持 `(0,10)` 一条；这不否定边表，但**R25 的"干净片段"结论要按此收缩**。

### 4.4 旧契约在真实实验里到底有没有触发（仪器审计）

方法：在**基线副本**里给 `track_path` 包一层记录（`audit_lag.py`）：记录每次 `links_at`/`newton_complex`
调用，按跟踪器自己的接受条件（`step <= max_step` 且 `residual < tolerance`）重建被接受的参数链，
再与请求的 `s_points` 按走行顺序逐步比对（两指针、精确命中判据 1e-9）。
包装器只记录、代调用真函数——透明性判据：同一次运行的实验输出与未插桩的基线逐字相同
（exp45 四个 fold 行、exp38 全部行一致）。

| 调用点 | 调用数 | 校正器试步 | 被拒（触发折半） | 请求点 | 精确命中 | 未命中（漏走） | ok=True |
|---|---|---|---|---|---|---|---|
| exp38 phase A.3 :182 | 6 | 902 | **0** | 902 | 902 | **0** | 6/6 |
| exp38 phase B :200 | 16 | 3316 | **0** | 3316 | 3316 | **0** | 16/16 |
| exp45 纤维 :79 | 6 | 902 | **0** | 902 | 902 | **0** | 6/6 |
| exp45 回路 :110 | 64 | 46445 | **5522** | 56272 | 29009 | **27263**（最坏一次 1113 点） | 44/64 |

读法：

* **exp38 的 22 次调用一次折半都没触发**（4218 个请求点全部精确命中）⇒ 旧契约在 exp38 这一轮里
  根本没有机会出错；它的"修复前 = 修复后"因此是**机制性**结论，不只是数字巧合。
* **exp45 的回路调用相反**：5522 次折半、27263 个请求点没走到（其中 **10380 个发生在 `ok=True`
  的调用里**，最坏一次跳过 1113 个点），却仍有 44 次返回 `ok=True` ⇒ 这正是 §4.3 读数变化的机制。
* 值得记住的细节：**两次实验里 `ok=True` 调用的最终参数都到达了**——旧代码的漏走发生在**中间段**，
  终点往往在下一段的第一步恰好被接受。所以"只看终点是否到达"抓不住这个缺陷，
  必须看整条链（这也是 §3 表里那几个"只信 ok + 最近根"的调用点危险的原因）。

> 仪器自检：同一统计量在**修复后**的跟踪器上跑同一实验应为"未命中 0"。结果（`exp45_lag_fixed.json`，
> 已用 `PYTHONPATH` + 模块路径确认跑的是修复后的 `study/exp38_monodromy_group.py`，日志首行打印了该路径）：
>
> | exp45 回路 :110 | 调用数 | ok=True | ok=True 的未命中 | ok=False 的未命中 |
> |---|---|---|---|---|
> | 修复前 | 64 | 44 | **10380**（最坏一次 1113 点） | 16883（20 次中途失败） |
> | 修复后 | 64 | 36 | **0** | 17539（28 次中途失败） |
>
> ⇒ 判据成立：`ok=True` 却漏走请求点只在旧契约下出现；修复后 `ok=True` ⇒ 逐点走完。
> 同时注意修复后 `ok=True` 的次数**变少**（44→36）：旧代码把"走不完"报成成功，新代码如实返回失败
> ——这正是 exp45 修复后 `untracked` 变多（8/6/4 → 10/8/6/4）的原因。

### 4.5 与 day06 记录表的差异（不是 T1 造成的）

记录（`day06-result.md` R15，基线提交 `278b7c4`）：7 fold、边
`(10,0) (13,7) (9,14) (1,8) (10,11) (12,13) (14,15)`、分量 `[3,3,3,2,1,1,1,1,1]`。
HEAD（修复前后都一样）：8 fold（多出 1.047164），边
`(10,0) (7,12) (15,9) (8,1) (10,11) (12,13) (14,15) (9,5)`、分量 `[4,3,3,2,1,1,1,1]`。

* 差别一：fold 1.047164 的边 `(9,5)`——kimi 后来补的第 11 条（`day11-result.md` R26），
  它把 `{9,14,15}` 与 `{5}` 并成 4 元分量，恰好解释分量从 9 个变 8 个。
* 差别二：两个诞生 fold 的**配对**不同（0.656878：`(7,12)` vs 记录 `(13,7)`；
  0.657375：`(15,9)` vs 记录 `(9,14)`），但**分量不变**（都是 `{7,12,13}`、`{9,14,15}`）。
  这可能是"根编号顺序变了"（普查顺序/BLAS）而不是分支结构变了。
* 注意：记录表写作时（`278b7c4`）`study/exp38_monodromy_group.py` **还没有进版本库**
  （首次提交是 44 分钟后的 `e3dbd0d`），所以"照记录提交复跑"只能取 `e3dbd0d`。

**归档复现（新增证据）**：用 `git archive e3dbd0d` 取出首次提交的 exp38 原样运行：

```bash
mkdir -p /home/yang/workspace/research/.t1_day06b
git archive e3dbd0d | tar -x -C /home/yang/workspace/research/.t1_day06b
cd /home/yang/workspace/research/.t1_day06b && python3 -m study.exp38_monodromy_group
```

结果（exit 0，6m15s）**与 day06 记录逐项相同**：

```
  7 folds: ['0.655086', '0.656878', '0.657375', '0.658119', '0.718354', '0.799244', '0.961848']
  fibre after adding complex roots: 16 (expect 16)
  fold 0.655086: transposition (10, 0)      fold 0.718354: transposition (10, 11)
  fold 0.656878: transposition (13, 7)      fold 0.799244: transposition (12, 13)
  fold 0.657375: transposition (9, 14)      fold 0.961848: transposition (14, 15)
  fold 0.658119: transposition (1, 8)
  7 edges on 16 roots -> components of sizes [3, 3, 3, 2, 1, 1, 1, 1, 1]
```

⇒ **记录表在它自己的代码版本上可复现**（不是伪造、也不是被 T1 缺陷污染的版本）；
HEAD 的差异（多出 fold 1.047164 的 `(9,5)`、两条配对改变、分量 `[4,3,3,2,1,1,1,1]`）
来自 `e3dbd0d` 之后的代码变化，而**不是** T1 修复——判据是 HEAD 修复前/后逐字相同。
"具体是哪次改动"记未决（见 §5 第 13 条，附 bisect 命令）。

## 5. 记录结论的分类

### 不受影响（有独立证据，或缺陷未触发）

1. **`exp44` 的群计算**（10 边 ⇒ `[6,3,3,1,1,1,1]`、阶 25920）：纯置换运算，不经过跟踪器；前后逐字相同。
   注意它只验证"给定边表 ⇒ 群"，**不验证边的来源**。
2. **`exp38` 的 16 根纤维构造**（10 实 + 6 复）：phase A.3 独立复核 s=1 的 12 分量残差；前后逐字相同。
3. **`exp37` 的 Q15 对换 `[1,0]`**（两旋向）：仪器审计证明修复前该调用 0 次折半、整圈走完；修复后同结果。
4. **`exp46` 的 `(5,9)` 边**（fold 1.047164）：前后逐字相同。
5. **`exp40` 在 400 种子下的复分支点集合与边**：前后逐字相同。
6. **全部实侧结论**（aspect / 唯一性域 / cusp / 规划 / SR0 单值化）：这些实验不使用 `track_path`/`track_circle`
   （§3 的同型检查：`monodromy.track_pose_path`、`homotopy.track`、`taskspace.fiber_lift`、`chamber`、
   `exp21/35/49/50/51/52/57/58/59` 都是"拒绝即重试同一步/返回失败"的结构）。
   抽查 `python3 -m study.monodromy --robot sr5`（13.6 s，exit 0）：读数与记录一致（各回路恒等、无证书）；
   并且 `study/monodromy.py` 的 import 表里**没有**被改的两个模块（`from study import homotopy` /
   `ik_structure` / `census`），所以它对 T1 是**结构无关**的，不需要 A/B。

### 有风险（结论依赖被缺陷覆盖的调用点）

7. **fold→对换边表与分量结构**（Q7/Q15）：调用点 `exp38` phase B、`exp40` phase 0/2、`exp45`、`exp46` 都只信
   `ok` + `1e-2` 最近根匹配。修复前后在 HEAD 上不变，但**与 day06 记录的两条配对不一致**（§4.5；
   记录表本身在其提交 `e3dbd0d` 上可复现），且 `exp45` 的读数确实被污染过（§4.3）
   ⇒ 需 T2 统一基点/根编号后重算，**在此之前两套编号不要混用**。
8. **`exp40` 复分支点的"数目"**（记录 14 或 6，本仓库 400 种子得 2）：与 T1 无关（前后相同），
   属于种子数与去重约定问题（T2 范围）。
9. **`exp45` 的四条对抗性复核边**：修复后只有 0.655086 的 `(0,10)` 可读（MATCH），另三条 untracked 8/6/4；
   插桩审计给出污染的量级（5522 次折半、27263 个请求点漏走、44 次 `ok=True`，§4.4）⇔ 修复前那四条读数
   **不可采信**。记录 R25 的"干净片段一致"应收缩为"fold 0.655086 的 `(0,10)` 已由修复后的仪器复现"，
   其余三条要等 Q17 的预测-校正内核。
10. **`exp38` phase B 的边在旧契约下可能被"错根匹配"**：匹配阈值 1e-2，而旧契约允许每个接受子步移动
    ≤ `max_step`（默认 0.1），原则上可落在另一条根的 1e-2 邻域内。插桩审计显示**本轮 16 次调用
    0 次折半**（§4.4），所以本轮没有实例；风险对"以后用更大 `max_step`/更小 `tolerance` 跑"依然成立。

### 已证伪

11. **"空圆负对照不动 / 双对照干净"（Q15 记录、`summary/03` §2.1）**：修复前 3/4 次调用
    **从未移动**（`accepted=0`、`refusals=8`、`|dq|=0`、`signal=inf`），打印的 `0.0e+00` 是"原样返回"而非"自返"；
    第 4 次与 fold 的纤维比较。⇒ 该对照当时是空洞的；结论"远离 fold 的圆回路不动"**当时未被测量**。
    修复后已实测：12/12 根在新基点纤维上自返（≤4.8e-13）。
12. **跟踪器契约本身**："`ok=True` ⇒ 到达终点"为假（探针 [1]–[5]、[8]–[10] 与 `exp45` 的 `(15,14)` 伪边）。

### 未决（附可结算命令）

13. **day06 记录表已在自己提交上复现**（§4.5），所以不是"记录写错"；但 HEAD 的两条配对与它不同，
    "是哪次 `e3dbd0d`→HEAD 之间的改动换掉了配对"未决，T1 已排除跟踪器契约（修复前/后逐字相同）。
    结算命令（需脚本化判据）：
    ```bash
    git bisect start HEAD e3dbd0d -- study/
    git bisect run python3 -m study.exp38_monodromy_group   # 判据：phase B 的 0.656878 行是 (13, 7) 还是 (7, 12)
    ```
    另一种可能（更便宜）：把两版的 s=1 十六根纤维按构型距离一一配对，看**根编号**是否只是一个置换；
    若是置换，则记录表与 HEAD 表是同一分支结构的两套编号，Q7 的"11 条边"必须**在同一编号下**重算（T2）。
14. `exp45` 三条读不出的边：需要 Q17 的预测-校正内核；本轮**不改**（不在 T1 范围）。
15. `exp40` 复分支点数目（记录 14 或 6，本仓库 400 种子得 2）：与 T1 无关（前后相同），
    需要按 T2 的"统一 2000 种子 + 同一去重/基点约定"重跑。

## 6. 验收对照与可复现命令

| T1 验收项 | 结果 | 证据 |
|---|---|---|
| `q(s)=s` 探针到达 1，或明确失败 | **到达 1** | §1.1 修复后 `[1.+0.j] 0.0 True` |
| 不再提前成功 | **满足** | `exp60` 11/11；含"最后一段需细分"与"校正器持续失败" |
| 折角多段折线 / 反向路径 / 负方向圆回路 / 最后一段细分 / 校正器持续失败 的回归检查 | **全部落地** | `exp60` 的 [1]–[11] |
| 检查绕圈的最终未展开角度（不只看 `exp(iθ)`） | **满足** | `exp60` [8]–[11]：+2π / −2π / 8 采样仍 +2π / 半圈时报 +π 且 `unfinished>0` |
| 显式最大细分/步数预算与最小步长；耗尽返回失败 | **满足** | `max_subdivisions`/`min_span`/`max_calls`；`exp60` [5][6][11] |
| 修复 exp37 负对照基点纤维 + 验证输入根满足起点方程 | **满足** | §2.3、§4.2 |
| 保持调用签名（或统一更新调用者） | **满足** | 三个返回值语义保持；`track_circle` 第三值语义变更且唯一调用者已同步 |
| 不改 `model/`、`tests/` | **满足** | `git status --short` 只有 `study/` |
| `ruff check .` / `pytest` | **通过** | `ruff`: All checks passed；`pytest -q`: **217 passed** |

```bash
# 探针与回归（修复后 11/11，退出码 0）
python3 -m study.exp60_tracker_contract

# 修复前后对照：先做修复前基线（HEAD = 修复前的提交），再在正式仓库跑同一批命令
mkdir -p /home/yang/workspace/research/.t1_baseline
git archive HEAD | tar -x -C /home/yang/workspace/research/.t1_baseline
cd /home/yang/workspace/research/.t1_baseline        # 修复前
python3 -m study.exp37_complex_loop
python3 -m study.exp38_monodromy_group
python3 -m study.exp40_complex_branches --seeds 400
python3 -m study.exp44_group_from_edges
python3 -m study.exp45_fold_loops
python3 -m study.exp46_strata
python3 -u audit_lag.py exp38_monodromy_group --out logs/exp38_lag2.json   # 6m01s，exit 0
python3 -u audit_lag.py exp45_fold_loops     --out logs/exp45_lag2.json   # 3m56s，exit 0
python3 -u audit_circle.py                                                # exp37：主回路 0 拒绝 / 负对照 3 次空转
cd /home/yang/workspace/research/py_robot_library    # 修复后
python3 -m study.exp37_complex_loop                  # 其余实验同上
python3 -m study.monodromy --robot sr5               # 结构无关抽查，13.6 s

# 修复后的同一统计（仪器自检）：脚本放到中性目录、用 PYTHONPATH 指向正式仓库
cp /home/yang/workspace/research/.t1_baseline/audit_lag.py /tmp/ && cd py_robot_library
PYTHONPATH=$PWD python3 -u /tmp/audit_lag.py exp45_fold_loops --out /tmp/exp45_fixed.json

# day06 记录表的归档复现（e3dbd0d = exp38 首次提交）
mkdir -p /home/yang/workspace/research/.t1_day06b
git archive e3dbd0d | tar -x -C /home/yang/workspace/research/.t1_day06b
cd /home/yang/workspace/research/.t1_day06b && python3 -m study.exp38_monodromy_group
```

原始日志与机器可读数据（本轮已归档）：`/home/yang/workspace/research/.t1_evidence/`
（`*_before.log` / `*_after.log` / `*_audit*.log` / `*_lag*.json.gz` / `audit_*.py`）。

## 7. 改动的文件

| 文件 | 改动 |
|---|---|
| `study/exp38_monodromy_group.py` | `track_path` 契约重写（子步迭代到终点 + 预算/地板 + 终点残差复核 + docstring） |
| `study/exp37_complex_loop.py` | `track_circle` 同上（按未展开角推进 + 第三返回值语义变更）；主回路调用者同步；输入根抛光与基点方程验证；负对照改用新基点纤维；docstring 更新 |
| `study/exp60_tracker_contract.py` | **新增**：GPT 探针 + 11 项回归检查 + 冻结的修复前控制流（A/B 对照） |
| `study/collab/QUESTIONS.md` | 登记 `exp60` 编号（T1） |

未改：`model/`、`tests/`、其他实验；无提交。
