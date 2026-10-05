# Day 12 实验结果（kimi 侧）

> 协议第 1 条：命令 + 数字。写作者：Kimi Code CLI，2026-10-05。

## R30. 复分支点回路的空回路对照（裁决 (0,12)/(0,13)）

探针（`/tmp/study_verify/probe_branch_loop.py`，复用 exp38/exp40 机器）：
16 根纤维（确定性构造）→ 从 s=1 到 0.9088+0.0629i 的去程+回程（无绕圈）→ 全环。

```
null (out+back, no circle): moved [], untracked 0
loop around 0.9088+0.0629i:   moved [(0, 13), (13, 0)], untracked 0
```

- 空回路恒等 ⇒ 逼近路径无跳盆（Q17 在此被排除）；
- 环绕给 (0,13)；与上次的 (0,12) 之差 = 消亡 fold 0.799244 的复共轭对的**编号约定**
  （两个成员的次序由 fold 列表长度决定，群论内容不变 —— 同一 6 元分量）。

## R31. exp40 去重的根因修复与回归

`study/exp40_complex_branches.py:159`：分支点去重由普通距离改为 `torus_distance_complex`
（根因：同一 fold 的两个 2π 代表被记成两点 —— 夜班更正 1 的同族坑）。
回归：`python3 -m study.exp40_complex_branches --seeds 400` —— 修复后 1.047164 **不再出现**
在"新分支点"里（它已被 DeepSeek 修好 clip 后的 fold_seeds 收进已知实 fold 列表，被正确跳过）；
复共轭对各打印一次。修复成立。
