# Day 5 实验结果（kimi 侧）

> 协议第 1 条：命令 + 数字。写作者：Kimi Code CLI，2026-10-03，基线提交 `362d19a`。
> 新仪器：`study/exp36_workspace_cusp.py`（s 固定的双坐标工作空间切片）、
> `study/exp37_complex_loop.py`（复平面绕 fold 的对换测量）。

## R13. cusp 狩猎的负结果汇总（七个 2D 切片，全部范围内为空）

命令：`python3 -m study.exp35_cusp --u-index {0,1,2} [--u-range ...]`、
`python3 -m study.exp35_cusp --u-kind rot --u-index {0,1,2} --u-range 0.6`、
`python3 -m study.exp36_workspace_cusp --crossings-from u0.npz --u-index 0 --v-index 1`

| 切片 | 非平凡汇合 | 单曲线回转 | 备注 |
|---|---|---|---|
| (s, x) ±0.15 m | 0 | 0 | fold 曲线行至 s∈[0.28, 1.41] |
| (s, y) ±0.25 m | 0 | 0 | |
| (s, z) ±0.25 m | 0 | 0 | 曲线近竖直（u 走 0.5 m） |
| (s, rot x/y/z) ±0.6 rad | 0 | 0 | 姿态三个切片同样为空 |
| (x, y) @ s=1 ±0.15 m | 0 | 0 | 7 个精确 fold 种子（u∈[0.085, 0.129]），14 条曲线 |

读法：判别式在 exp21 位姿附近的这些切片里是**互不相交的光滑 fold 弧**；
cusp 轨迹（参数空间余维 2）不在覆盖范围内。**"SR5 无 cusp"不能由此断言**
（cuspidality 已被 14 对见证路径直接证明；cusp 只是充分机制，且可以在别的区域）。

## R14. 复环回路：绕一个 fold 的对换 —— 本课题第一个非恒等单值化测量 ✅

命令：`python3 -m study.exp37_complex_loop`（绕 fold s* = 0.718354，复平面圆半径 1e-3，
96 采样，复数牛顿跟踪合并根对 + 两个远根对照 + 空圆负对照）

**主结果（两个旋向，逐位一致）**：

```
loop: centre s*=0.718354, radius 0.001, orientation + (base at angle pi, s0=0.717354)
  root 0 (pair): -> root 1 (distance 2.6e-10, |imag| 7.7e-18, worst residual 1.3e-14)
  root 1 (pair): -> root 0 (distance 9.1e-10, |imag| 8.7e-18, worst residual 1.4e-14)
  root 2 (control): -> root 2 (distance 4.4e-06, |imag| 5.8e-16)
  root 3 (control): -> root 3 (distance 8.9e-06, |imag| 1.3e-19)
  permutation [1, 0, 2, 3]: TRANSPOSITION
```

orientation − 同样给出 `[1, 0, 2, 3]`（单 fold 的两种旋向给出同一对换，如预期）；
**负对照**（同半径、圆心 0.85 附近无 fold）：全部根回到自身（0.0e+00 / 6.7e-02）。

结论：**绕一个 fold 的复闭环把两个合并根互换（对换），远根不动** ——
"每个合并点恰好两支"（此前只有分裂指数 ≈ 1/2 的证据）得到**拓扑独立证实**；
这是本课题测到的第一个非恒等单值化，也是复单值化群（Q7）的第一个生成元。

仪器记录（表现/根因/修法，按约定，共四条）：

1. `solve_lm` 从 fold 附近的任意种子**全部不收敛**（六个核方向种子无一成功）；
   根因：LM 求解器在近奇异处失败；修法：`homotopy.correct`（伪逆牛顿）从核方向种子
   稳定落在两根上（0.0669 / 0.0723 rad，与 exp21 的平方根 decade 表逐位一致）。
2. `se3_error` 与 `fk_links` 不能用于复数（arccos + float 强转）；
   修法：代数残差（位置 + 旋转矩阵共 12 分量）+ 复数最小二乘牛顿。
3. `links_at` 把复参数写进**实数数组**，虚部被静默丢弃、回路塌缩到实轴上
   （`ComplexWarning` 是唯一线索）；修法：`astype(complex)`。
4. 回路起点角默认 0 ⇒ 实际起点是 s*+ρ，而合并根对在 s*−ρ 侧才是实的 ——
   **基点与纤维不同侧**，第一步必然跳盆（worst residual 0.0 = 第一步就死）；
   修法：起点角按根对的实侧取（0 或 π），并加自适应半步（|Δq|>0.1 或残差超标则折半）。
