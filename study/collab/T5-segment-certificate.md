# T5 结果：把「这一段没有奇异」从采样断言升级为整段证书

> 日期：2026-10-17。执行者：DeepSeek Harness（teammate `certified-gap`）。任务书：
> [`GPT_MATH_ADVANCE_20261016.md`](GPT_MATH_ADVANCE_20261016.md) §B（B1–B4）；编号登记见
> [`QUESTIONS.md`](QUESTIONS.md)（T5 → `exp63_segment_certificate.py`，先登记再用）。
> 分支 `study/ik-branch-and-continuation`，**未提交**，改动留在工作区。
> 环境：Python 3.10.12 / numpy 2.2.6；命令均在仓库根目录执行；计时用于量级参考（同机有并行任务）。
> 新增/改动文件：`study/exp63_segment_certificate.py`（新）、`study/chamber.py`（**只加 opt-in，默认行为与旧数字不变**）。

## 0. 结论摘要

1. **界 `L` 按 §B1–B3 原式实现**（没有另造公式）：`ell_j` 由 `Ms` 布局的固定平移长度和给出，
   `a_j`/`b_j`/`L = sqrt(Σ_j α²b_j² + β²a_j²)` 逐字对应 B1/B3；只做位置任务时按 B3 删去角速度项。
   §B4 的浮点警告被当作**一等公民**：每个采样点用「近似逆 + 膨胀残差」给**严格下界**，
   库自身的 FK/Jacobian 求值误差另记一项 **numerical allowance**，所以证书的标签是
   **「解析界 + 浮点验证」，`machine_verified = False`**（区间后端是 T5 第二阶段，未做）。
2. **接口** `certify_segment(model, q0, d, ...) -> SegmentCertificate`：三态 `certified / inconclusive / invalid`，
   连同 `h`（实际用到的覆盖半径）、样本数、`L`、解析余量 `L*h`、数值余量、以及最小值的
   `bracket = [certified_lower, sampled_min]`。可用于**库里的任何模型**（6×6 几何 Jacobian、6×7 冗余臂、
   3×3 位置任务；§4 phase 3b 实测三种形状）。
3. **可失败自测 27 项全过**，退出码 0（`--quick --skip-paths` 为 20 项全过）。关键的一项：构造一条
   **采样看不见**的穿越段——9 点均匀采样的最小 `sigma_min = 1.319e-03 > 0`，项目自己的逐步检查器
   `chamber._segment_clears(..., delta=1e-3, density=9)` 也说 **clear**，而密采样 20001 点的最小是
   **1.218e-17**（|det J| 二分到 2.15e-17，`det J` 变号）——**证书拒绝**它（`inconclusive`，下界 0）。
   负对照（两端 `det J` 异号的随机段）4/4 全部拒绝。
4. **记录见证路径复核**（这就是「哪些结论升级」的答案，细节在 §6）：
   * **升级**：`exp49`（SR5，pose-seed 0）阶段 2 的见证路径——记录的 `fibre 10 (5+/5−)`、
     同 aspect 对 `(1,6)`、清除度 **5.25e-3**（`NOTES.md` 表）——**118/118 段全部认证**，
     整条折线 `sigma_min >= 4.983e-3`。⇒「这两个解在不碰奇异的路径上连通（同 aspect）」这条
     从**采样断言升级为整段定理**（在解析界 + 浮点标注的意义下）。
   * **未升级，而且被查出更糟**：`exp48` 的 3R 见证路线（记录 `smallest measured clearance 7.9e-5 ≈ 8e-5`，
     即那段「hug Σ」的**原始网格路线**）——证书在 2448 段里**拒绝 1 段**（子段 1258），
     且该段 `det J` **变号**（−2.63e-3 → +3.02e-4）、采样 `sigma_min` 从 6.89e-4 掉到 7.93e-5、
     中点 3.05e-4 ⇒ **该路线在两个网格节点之间穿过 Σ**。exp48 自己的 `NOTE`/docstring 已把这条
     路线记为「instrument issue（NOTES §3.33）」；本证书把它从「hug Σ」加强为「**穿 Σ**」。
   * **未复核**：exp49 的 pose-seed 1/2、14 对「不同 chamber」负向结论、`chamber` 的全部历史读数——
     **保持采样**（证书是 opt-in，不改任何默认行为）。
5. **成本**（同一台机器）：exp49 见证路径 118 段 / **5323 次 Jacobian / 0.72 s**；
   exp48 路线 2448 段 / **9751 次 / 2.26 s**；整轮实验实测 **1 min 29 s**（`time`，`user 1m30.8s`），
   其中 **~82 s** 花在**按 exp48/exp49 原样重建**那两条路径上（证书本身只占 ~3 s）。

## 1. 实现的界（§B1–B3，逐字对应）

记号到本库的映射（`Ms = [base, link_1, ..., link_n, tool]`，关节 `j` 绕
`Ms[0] @ Ms[1] @ Rz(q_1) @ ... @ Ms[j] @ Rz(q_j)` 的 `z` 轴转，工具点在 `Ms[-1]` 上）：

```
ell_j = Σ_{k=j+2..n+1} ||translation(Ms[k])||_2      # ≥ ||p - o_j||，只用到固定平移
a_j   = Σ_{k<j} |d_k|                                 # 上游旋转预算
b_j   = a_j * ell_j + Σ_{k≥j} |d_k| * ell_k           # ||dJv_j/dt||_2 ≤ b_j
L     = sqrt( Σ_j ( α² b_j² + β² a_j² ) )             # ≥ ||dJbar/dt||_F ≥ ||dJbar/dt||_2
L1    = Σ_j ( α b_j + β a_j )                         # B3 的免平方根（更保守）版本，只作参考
```

* `Ms[j]` 自己的平移**不计入**：它已经是关节 `j` 的原点；base 的固定旋转也不影响界（B1）。
* 只做位置任务（3R）时按 B3 「删去角速度项」：`L = sqrt(Σ_j α² b_j²)`，`rows=('v','v','v')`。
* `a_j, b_j` 对 `|d|` 线性 ⇒ 子段 `[t1,t2]` 的界就是 `(t2−t1)·L`，这正是自适应细分有效的原因。

SR5 实测（phase 1，`d` 是随机方向、`|d| = 0.5315 rad`）：

| 量 | 值 |
|---|---|
| `ell_j` (m) | 1.3737, 1.0457, 0.6426, 0.2395, 0.1035, 0.0 |
| `a_j` | 0.0, 0.0822, 0.2203, 0.4957, 0.7858, 0.9738 |
| `b_j` | 0.5232, 0.4963, 0.4075, 0.2077, 0.1008, 0.0 |
| `L` / `L1` | **1.6144** / 4.2932 |
| 有限差分 `‖dJ/dt‖₂/L` 的最坏比 | **0.251** |

有限差分那一行只是**实现索引检查**（B4 明说它不构成定理证明）：沿段取 48 个点做中心差分，
最坏只到 `L` 的 25%，说明 `L` 的索引没写错且界是松的（松是安全的）。

## 2. 接口

```python
from study.exp63_segment_certificate import certify_segment, certify_path

cert = certify_segment(model, q0, d, delta=5e-3)          # 整段 q0 + t*d, t∈[0,1]
cert = certify_path(model, walk.path, delta=5e-3)          # 折线：逐段认证
cert = certify_segment(three_r, q0, d, task=PositionTask(three_r), rows="auto")  # 位置任务
```

`SegmentCertificate` 的字段（`study/exp63_segment_certificate.py`）：

| 字段 | 含义 |
|---|---|
| `status` | `certified` / `inconclusive` / `invalid`（越限） |
| `certified_lower` | 整段 `sigma_min(J)` 的**认证下界**（未认证时为 0，永远不是一个越界的数） |
| `positive`, `delta_ok` | 是否 `> 0`；给了 `delta` 时是否 `> delta` |
| `bracket` | `[lower, sampled_min]`，真最小值落在其中（到余量精度） |
| `sampled_min` | 采样点上的浮点 SVD 最小值（**旧仪器的读数口径**） |
| `l`, `h`, `span`, `samples` | 界 `L`、binding 区间的覆盖半径、段长（rad）、Jacobian 求值次数 |
| `margin_lipschitz`, `margin_numeric`, `sample_deficit` | `L*h`、求值浮点余量、严格下界相对浮点 SVD 的损失 |
| `det_sign_change` | 采样 `det J` 是否变号（同一条独立的穿越见证） |
| `lower_jbar` | 换算到 `Jbar` 之前的界（单位不变性用它比对） |
| `machine_verified=False`, `stop`, `note` | 诚实标签；停止原因（`converged/collapsed/budget/uniform/zero`） |

约定：

* **只报下界，不报猜测**：`inconclusive` 时 `certified_lower = 0`，且**绝不**翻成「穿奇异」——
  这正是 B4 的要求。要谈穿越，必须另给独立见证（`det_sign_change` / 密采样 / 二分得到 |det J|）。
* `invalid` 用于**提升越限**：`q(t)` 按字面直线（不绕环面短路）检查 `[lower, upper]`，越限即 `invalid`
  （B4：「必须保存实际选定的连续角度提升，并检查该提升是否符合物理限位」）。
* 自适应细分的停止条件：先满足「所有区间的界 > target」，再把 bracket 收到
  `max(abs_tol, rel_tol*sampled_min) + sample_deficit`；binding 区间若因浮点分辨率无法再分
  （穿越/相切处必然发生），标记 `collapsed` 并停止细分而不是空转。

## 3. 数值部分：两条独立的严格路线（§B4 及其加强）

采样点的下界取**两条严格路线的最大值**（都是下界，取大仍严格）：

1. **B4 原路线**：取数值逆 `B`，`e = ‖I − BJ‖₂ < 1`、`M = ‖B‖₂` ⇒ `sigma_min(J) ≥ (1−e)/M`。
   方阵直接取逆，长方阵用较小一侧的 Gram 矩阵（再补一项 Gram 乘积自身的舍入）。范数用
   Frobenius 与 `sqrt(‖·‖₁‖·‖∞)` 的**较小者**（两者都是谱范数的上界）。实测这条路**偏保守 24.8%**：
   SR5 某构型 `sigma_min` 浮点值 0.08152281，B4 路线只给 **0.06134979**（`‖B‖_F = 16.30` 而
   `‖B‖₂ = 12.27`，因为 `B` 有若干个不可忽略的奇异值）。
2. **SVD 因子路线（本模块新增，仍然只用可验证的残差）**：对**高型**矩阵 `A`（宽型取转置，
   `sigma_min(Aᵀ) = sigma_min(A)`，且只有高型才有正交的右因子），取 `A ≈ U diag(s) Vᵀ`，
   在浮点里**验证**三个残差 `δ_R ≥ ‖U diag(s)Vᵀ − A‖₂`、`δ_U ≥ ‖UᵀU − I‖₂`、`δ_V ≥ ‖VᵀV − I‖₂`，
   于是 `sigma_min(A) ≥ s_min·(1 − (δ_U+δ_V)/2) − δ_R`。它不依赖 SVD 算法自身的后向稳定性，
   只依赖上面三个可计算的残差；实测损失 `2.2e-14`（相对），把上一条的 24.8% 保守性去掉了。
   两条路线都保留、都打印，证书用较大的那个。

**求值余量** `numerical_allowance(model, row_kinds, α, β)`：库自己算 FK/Jacobian 也有舍入。
用 Higham 的 `γ_k`，每个 4×4 乘积的误差量级是 `γ₄` 乘「累积平移的量级」，而累积平移被
`1 + Σ_k ‖translation(Ms_k)‖₂`（一个**长度**）界住——不是「各矩阵范数之积」，后者是相对误差界，
对长链病态地悲观。旋转行不带长度，误差按单位向量算；再乘 `8`（叉乘与索引开销）与 libm 项。
SR5/SR0 上这项是 **~6e-13**，比解析余量 `L*h`（~1e-3）小 10 个量级。它是**显式常数下的宽松估计**，
不是区间界——这是本实现与「机器认证」之间的距离，第二阶段（区间后端）才能补上。

`sample_deficit` 也一并报告：它是「严格下界 vs 浮点 SVD」的差，SR5 上 `2.2e-14`，
旧（纯 B4 路线）实现下曾达 `2.1e-2`，正是它让自适应细分永远收不紧（已修：停止容差里含 deficit）。

## 4. 自测结果（`python3 -m study.exp63_segment_certificate`）

全部 **27/27** 项检查（含 phase 1–3b 与 phase 4 的记录路径复核）**通过**，退出码 **0**。
摘录可读的证据行：

**phase 2a（成功例）**：SR5 上一段 `|d| = 0.0775 rad` 的段，`L = 0.2854`、`h = 7.81e-3`、51 次求值 ⇒
`certified sigma_min >= 3.671e-02`，bracket `[3.671e-02, 3.857e-02]`，
密采样 2001 点最小 `3.857e-02` ⇒ 证书**没有越过**密采样（一致性检查）。

**phase 2b（采样会误证的构造反例）**：

| 量 | 值 |
|---|---|
| 穿越点（对 `det J` 二分） | `|det J| = 2.15e-17`，位于段参数 `t = 0.55` |
| 9 点均匀采样的最小值 | **1.319e-03 > 0**（若把它当证书 ⇒ 误证） |
| `chamber._segment_clears(delta=1e-3, density=9)` | **True**（项目自己的逐步检查器也被骗过） |
| 密采样 20001 点最小 | **1.218e-17**（穿越是真的） |
| 本证书 | `inconclusive`，`certified_lower = 0`，bracket `[−5.994e-13, 0.0]`，3300 次求值 |

**phase 2c（异号端点负对照）**：随机取 4 条 `det J` 两端异号的段，**全部 `inconclusive`**，
最高 `certified_lower = 0`，且 4/4 都给出采样符号变化。

**phase 2d**：把段推进到关节限位之外 ⇒ `status = invalid`（不是 `certified`）。

**phase 3（3R 位置任务 + 单位不变性）**：`L = 0.5134`，`certified >= 2.9243e-01`，
bracket `[2.9243e-01, 3.0765e-01]`，密采样 `3.0765e-01`，20 次求值。

**phase 3b（接口对其它模型）**：`aubo_c5` 6×6、`er3` 6×7、`franka` 6×7 全部 `certified`
（下界 2.9971e-02 / 2.2286e-02 / 7.4987e-02；`L` 0.1751 / 0.2377 / 0.2184）。

**phase 4c（opt-in 不改默认）**：在 exp49 的见证端点（贪婪走法已知能到达）上，
默认 `clearance_path` 与包装版**逐字段相同**：`connected True ('arrived (final segment
verified)')`、`min_sigma 5.245e-03`、119 点；新字段 `ClearanceWalk.certified` 默认 `False`；
包装版给出 **118/118 段认证**、`lower 4.983e-03`，且 `walk.certified = True` 与
`walk.connected and certificate.certified` 逐位一致。

## 5. 单位缩放（B3 的 caveat）与不变性

几何 Jacobian 的行不同量纲：前 3 行是「长度/弧度」，后 3 行是「弧度/弧度」，所以
`sigma_min(J)` **依赖长度单位**。本实现的约定：

* 证书在**缩放矩阵** `Jbar = diag(α I₃, β I₃) J` 上做，`α = 1`（每**模型长度单位**，本库是米）、`β = 1`（无量纲）；
* 报出的 `certified_lower` 是**未缩放** `J` 的下界：`sigma_min(J) ≥ sigma_min(Jbar)/max(α,β)`
  （`α = β` 时取等，默认就是这种情况）；
* **不变性 1**（自适应模式）：`(α,β) → (tα, tβ)` 不改变报出的界（实测：`lower_jbar`
  0.29242619 → 0.58485237，除以 `max` 后两者**逐位相同** `0.2924261859`）。
* **不变性 2**：把整个 3R 的长度 ×1000（换成毫米）并同时取 `α → α/1000`，
  `Jbar` 层面的界**不变**（固定 17 点网格：0.29161046186 vs 0.29161046186，相对差 **3.18e-13**；
  `L` 也逐位不变）。这是「换长度单位」的正确语义：`α·b_j` 与 `β·a_j` 各自不变。
* 自适应模式的停止判据会在舍入处左右横跳（实测自适应差 9.33e-14，远小于 bracket 宽度 1.52e-02），
  所以不变性检查用固定 `uniform=17` 网格；两种模式的数都是合法下界。

## 6. 记录见证路径：sampled vs certified

复现方式：`exp63` 里**逐字重放**两条路径的构造（同随机序列、同预算、同调用顺序），
因此下表的「旧采样最小值」应与记录一致（`NOTES.md` 的 exp49 表、exp48 的 phase 4 打印）。
exp49 的重放按 README/Q22 的记录命令取 `--least 8 --tries 14`（模块 argparse 默认是 10/12，
**不**是记录用的那组）；重放结果与 `NOTES.md` 的 seed 0 行逐项相同，说明用的就是那条路径。

| 路径 | 记录读数（旧，采样） | 证书自己的采样最小 | **认证下界** | 段数（认证） | 成本 |
|---|---|---|---|---|---|
| exp49 SR5，pose-seed 0，对 `(1,6)`，`--least 8 --tries 14` | `5.25e-3`（`NOTES.md` 表逐字） | 5.2448e-3 | **4.9829e-3** | 118/118 | 5323 次 / 0.70 s |
| exp48 3R 见证路线（300² 标签网格 + 原样回退到原始网格路线） | `7.9e-5`（记录 8e-5，"hug Σ"） | 0.0（子段 1258） | **0（拒绝）** | 2447/2448 | 9751 次 / 2.22 s |

明细：

* **exp49（SR5）**：重放得到 `10` 个解、同号对 `(1,6)`、walk `reason = 'arrived (final segment
  verified)'`，与 `NOTES.md`「seed 0：fibre 10 (5+/5−)，对 (1,6)，清除度 5.25e-3」逐项吻合
  （`NOTES.md` 另有「沿一条关节路径清除度停在 5.25e-3」的独立记录）。
  证书：`L_max = 0.3648`、`h_worst = 1.95e-3`、数值余量 `5.99e-13`，**118/118 段 certified**，
  **0** 次 `det J` 变号 ⇒ 整条折线 `sigma_min ≥ 4.9829e-3`。
  注意 `delta = 5e-3` **没有**被认证（4.9829e-3 < 5e-3）：可以认证「> 0（同 aspect）」，
  但不能认证「≥ 5e-3」——这是 bracket 5% 精度的诚实后果，不是矛盾。
* **exp48（3R）**：重放同 `rng` 序列后，`clearance_path` 的细化腿**失败**（与 exp48 的
  `waypoint leg failed` 分支一致），于是走**原始网格路线**（2449 点，最大步长 4.9e-3 rad）。
  记录只检查了**节点**的 `sigma_min`（`min over route nodes = 7.93e-5`；exp48 源码注释记的正是
  「unhugged grid route passed within 8e-5 rad of Sigma」），`aspect_path` 的 docstring 也明说
  「网格只保证节点是清楚的，边要靠调用方密采样验证」。
  证书检查**整段**，在子段 1258 找到：

  | `t` | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
  |---|---|---|---|---|---|
  | `sigma_min` | 6.89e-04 | 4.97e-04 | 3.05e-04 | 1.13e-04 | 7.93e-05 |
  | `det J` | −2.63e-03 | −1.89e-03 | −1.16e-03 | −4.30e-04 | **+3.02e-04** |

  ⇒ 该子段内部 `det J` 变号（区间套 + 连续性），**路线穿过 Σ**；证书对此段 `inconclusive`、
  整条路线 `certified = False`。这正是 T5 想抓的那种「节点清楚、中间穿过去」的漏洞，
  而且在**本项目已经记录的路线**上真实发生了。

## 7. 升级 / 仍为采样 / 未决

**升级（sampled → certified）**

| # | 结论 | 依据 |
|---|---|---|
| 1 | exp49 阶段 2（pose-seed 0）：`(1,6)` 两解在同一 aspect（存在不碰 Σ 的路径） | 118/118 段认证，整条折线 `sigma_min ≥ 4.983e-3` |
| 2 | 同一路径「最坏清除度 5.25e-3」这个读数 | 认证为 `≥ 4.983e-3`（bracket 5% 内），并且**不再依赖采样密度** |
| 3 | `chamber` 的逐步检查 | 新增 opt-in `chamber.certified_clearance_path`，把「走到且每步采样合格」升级为「整条折线认证」；默认行为不变 |

**仍未升级（保持采样）**

| # | 结论 | 说明 |
|---|---|---|
| 4 | exp48 阶段 4「路线非奇异（最小 8e-5 > 0）」 | **撤回**：证书拒绝，且给出穿 Σ 的独立见证（`det J` 变号） |
| 5 | exp49 pose-seed 1/2 的见证路径、14 对「不同 chamber」负向结论、exp02/exp06 的历史读数 | 未跑证书；「不同 chamber」的**单向**证书（`det J` 同号/异号）本来就不是「不碰奇异」型断言，不受影响 |
| 6 | `chamber` 所有历史调用与 `study/` 其它模块的默认路径 | 证书是 opt-in；`clearance_path` 的签名、数字、随机消耗**逐字未动** |

**未决（不猜）**

| # | 事项 |
|---|---|
| 7 | **机器认证**：区间后端的 FK/Jacobian（T5 第二阶段）未做；当前 `machine_verified = False`，数值余量是显式常数的宽松估计 |
| 8 | exp48 子段 1258 的穿越是**横截**还是**相切**：`det J` 变号说明横截，但没有对该穿越点做 Newton 精化（证书对两种情形都拒绝，结论不变） |
| 9 | exp49 的 seed 1/2 与 exp48「腿成功」的细化路线分支在本轮**不可达**（腿失败是记录行为），故未认证 |
| 10 | 证书在**参数延拓/复**方向不适用（B3 明确：复旋转不再是实正交等距变换） |

## 8. 复现

```bash
cd py_robot_library
python3 -m study.exp63_segment_certificate                     # 全部检查，退出码 0；1 min 29 s
python3 -m study.exp63_segment_certificate --quick --skip-paths # 20 项快检，~20 s
python3 -m study.exp63_segment_certificate --json out.json      # 机器可读记录
ruff check study/exp63_segment_certificate.py study/chamber.py
python3 -m pytest -q                                            # 217 passed（未改 model/、tests/）
```

本轮运行：退出码 **0**，**27/27** 检查通过（`--quick --skip-paths` 为 20/20）。JSON 记录：
`exp63_full.json`（命令、`L`、每个自测的通过情况、两条路径与 phase 3b 的逐项读数）。

## 9. 与任务书 §B/§T5 的逐条对照

| 任务书要求 | 落地 |
|---|---|
| B1 记号、由连杆长度给 `ell_j` | `chain_lengths(model)`，只用 `Ms` 的固定平移范数 |
| B2 证明给出的 `a_j, b_j` | `lipschitz_bound`，逐字实现；附有限差分**索引**对照 |
| B3 `L`、`L1`、整段认证 `min_k s_k − L·h`、只做位置任务时删角速度项 | `lipschitz_bound` + `certify_segment`（`rows`）；`L1` 一并报告 |
| B4 浮点警告：SVD 不能直接当严格下界、要膨胀残差 | `sigma_min_sample_bounds`：B4 近似逆路线 **+** SVD 因子残差路线（取大者）；余量显式 |
| 三态 `certified / inconclusive / invalid`，失败不自动断言穿奇异 | `SegmentCertificate.status`；`note` 明说不作断言 |
| 自适应细分、限位检查 | 二分细分 + bracket 停止规则；`check_bounds` 按**字面提升**检查 |
| 不修改库 | 未改 `model/`、`tests/`；`study/chamber.py` 只加 opt-in 函数与一个默认 `False` 的字段 |
| 拿同号见证路径复核、异号端点负对照不得通过 | §6 两条路径 + phase 2c 4/4 拒绝 + phase 2b 的误证反例 |
