# 复现抽查（基线提交 9e15978）

本文件记录在提交 `9e15978` 上重跑关键命令时的实际输出（每个命令截取末尾 6 行），
用于确认 README/NOTES 里“命令 → 测量”的链条在最终状态下仍然成立。
重新生成：`bash study/reproduce.sh`（约 5~8 分钟）。

## `python3 -m study.exp10_sr0_geometry`

```
  link 5 y = +0.1360 -> 0.0:  (np.float64(0.0), np.float64(0.0), np.float64(0.0))  CLOSES THE WRIST

SR0 = SR5 with that component zeroed:
  wrist triples: (np.float64(0.0), np.float64(0.0), np.float64(0.0))
  all pairwise intersections: ['1-2', '1-4', '4-5', '4-6', '5-6']
  travelled distance: link 5 y +0.1360 -> 0.0 m
```

## `python3 -m study.exp12_sr0_solver --poses 2 --seeds 300 --match 1e-3`

```

analytic counts [8, 8]
numerical counts [14, 8]
worst analytic pose residual: 8.01e-10
analytic time per solve: 66.7 ms
totals: census-only solutions 0, analytic-only solutions 0
```

## `python3 -m study.exp21_endgame --pose-seed 0`

```
pose seed 0; folds are computed, not walked around

  seed 2: stalled s = 0.7173 -> fold s* = 0.718354 (augmented residual 3.2e-11) | solutions within 0.2 rad: 6, nearest partner 4.20e-03 rad | distances per decade [0.1971, 0.06692, 0.0223, 0.00972, 0.00724] | split exponent 0.440 over 4 points (square root: two branches)
  seed 4: stalled s = 0.9607 -> fold s* = 0.961848 (augmented residual 9.0e-13) | solutions within 0.2 rad: 5, nearest partner 5.83e-04 rad | distances per decade [0.11523, 0.03768, 0.01366, 0.00816, 0.00736] | split exponent 0.389 over 4 points (check)
  seed 6: stalled s = 0.7972 -> fold s* = 0.799244 (augmented residual 6.5e-11) | solutions within 0.2 rad: 7, nearest partner 5.63e-03 rad | distances per decade [0.19116, 0.06783, 0.02294, 0.01071, 0.00849] | split exponent 0.423 over 4 points (square root: two branches)
```

## `python3 -m study.exp25_deterministic_count --pose-seed 0`

```
pose seed 0: start with 8 solutions at s = 0 (SR0's fibre)
  window [0.650, 0.662] with 4 fold(s): fibre 8 -> 16 (delta +8, predicted +8 in magnitude) | survivors 8/8 carried, 0 died, 8 born
  window [0.710, 0.725] with 1 fold(s): fibre 16 -> 14 (delta -2, predicted +2 in magnitude) | survivors 14/16 carried, 2 died, 0 born
  window [0.792, 0.806] with 1 fold(s): fibre 14 -> 12 (delta -2, predicted +2 in magnitude) | survivors 12/14 carried, 2 died, 0 born
  window [0.950, 0.975] with 2 fold(s): fibre 12 -> 10 (delta -2, predicted +4 in magnitude) | survivors 10/12 carried, 2 died, 0 born
```
