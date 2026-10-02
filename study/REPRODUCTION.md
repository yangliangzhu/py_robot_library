# 复现抽查（基线提交 64d6c05）

本文件记录在提交 `64d6c05` 上重跑本课题十条头条命令时的实际输出（每个命令截取末尾若干行），
用于确认 REPORT.md / README.md / NOTES.md 里“命令 → 测量”的链条在最终状态下仍然成立。
重新生成：`bash study/reproduce.sh`（约 15 分钟）。

## `python3 -m study.exp10_sr0_geometry`

```

exhaustive single-parameter relaxation (all 6 links x 3 translation components), wrist triples in metres:
  link 5 y = +0.1360 -> 0.0:  (np.float64(0.0), np.float64(0.0), np.float64(0.0))  CLOSES THE WRIST

SR0 = SR5 with that component zeroed:
  wrist triples: (np.float64(0.0), np.float64(0.0), np.float64(0.0))
  all pairwise intersections: ['1-2', '1-4', '4-5', '4-6', '5-6']
  travelled distance: link 5 y +0.1360 -> 0.0 m
```

## `python3 -m study.exp11_dh_continuation --steps 3`

```
   step (m)   first order    +1 Newton    +3 Newton     exact IK  |q3 - q_exact|    ratio
    0.10000     1.721e-02    2.926e-03    1.075e-11    4.061e-02        1.93e-01        -  (IK did not converge)
    0.01000     1.712e-04    1.809e-07    1.545e-14    3.076e-08        4.00e-08    100.5
    0.00100     1.711e-06    7.518e-12    1.422e-16    1.241e-06        1.51e-06    100.1

the same step with J_q ill-conditioned (solution driven towards the singular set):
  sigma_min 2.96e-03: |dq| = 2.204 rad, first-order error 6.966e-01 m, after 3 Newton steps 2.017e-02 m
  (compare with the well-conditioned case above at the same step)
```

## `python3 -m study.exp12_sr0_solver --poses 2 --seeds 300 --match 1e-3`

```
pose 0: analytic 8, numerical 14 (from 146/300 converged) | numerical solutions the analytic solver missed: 0 | analytic solutions the census missed: 0 | generated configuration recovered: True
pose 1: analytic 8, numerical 8 (from 181/300 converged) | numerical solutions the analytic solver missed: 0 | analytic solutions the census missed: 0 | generated configuration recovered: True

analytic counts [8, 8]
numerical counts [14, 8]
worst analytic pose residual: 8.01e-10
analytic time per solve: 67.7 ms
totals: census-only solutions 0, analytic-only solutions 0
```

## `python3 -m study.exp02_pair_census --poses 2 --seeds 300 --resolve`

```
  pose 0 pair (1,6): unresolved at 89% of the way (no admissible step, min clearance 5.001e-03)
  pose 0 pair (1,7): unresolved at 59% of the way (step budget exhausted, min clearance 5.031e-03)
  pose 0 pair (3,5): unresolved at 55% of the way (no admissible step, min clearance 5.001e-03)
  pose 0 pair (4,5): unresolved at 66% of the way (step budget exhausted, min clearance 5.025e-03)
  pose 0 pair (6,7): unresolved at 63% of the way (step budget exhausted, min clearance 5.001e-03)
  connected while keeping clearance (same chamber): 4
  not connected (unresolved or different)         : 27
  negative control -- certified-different pairs the walk could not connect: 18/18
```

## `python3 -m study.exp15_branch_labels --poses 1 --seeds 400`

```
    solution 3: det sign -1, labels (shoulder, elbow, wrist) (-1, 1, -1)
    solution 4: det sign -1, labels (shoulder, elbow, wrist) (-1, -1, -1)
    solution 5: det sign -1, labels (shoulder, elbow, wrist) (1, 1, -1)
    solution 6: det sign +1, labels (shoulder, elbow, wrist) (-1, 1, -1)
    solution 7: det sign +1, labels (shoulder, elbow, wrist) (1, -1, -1)

label agreement: same-label pairs connected 0, same-label pairs NOT connected 4, different-label pairs connected 7
sign-certificate violations by walks: 0
```

## `python3 -m study.exp19_partition --poses 1 --seeds 600 --waypoints 4`

```
pose 0: 8 solutions (206/600 seeds), sign groups 4+/4-
  pairs 28: 8 certified same branch, 16 certified different (sign), 4 undetermined
  after merging the witnessed pairs: 2 groups (sizes [4, 4]), each inside one sign group
  of the undetermined pairs, 3 show a dip below 0.01 (the heuristic would call them different branches; its measured error rates are in exp02)
```

## `python3 -m study.exp20_labels_redundant --robot franka --seeds 120`

```
  solution 0: labels along the self-motion path (1, 1, -1) | travelled 12.00 rad, min clearance 1.06e-01, pose drift 5.5e-14 | labels constant
  solution 1: labels along the self-motion path (-1, 1, -1) -> (-1, 1, 1) | travelled 12.00 rad, min clearance 1.56e-01, pose drift 2.6e-15 | LABELS CHANGE
  solution 2: labels along the self-motion path (-1, 1, -1) -> (-1, 1, 1) | travelled 12.00 rad, min clearance 8.02e-02, pose drift 6.8e-13 | LABELS CHANGE
  solution 3: labels along the self-motion path (1, 1, -1) -> (1, 1, 1) | travelled 12.00 rad, min clearance 9.48e-02, pose drift 6.0e-15 | LABELS CHANGE
  solution 4: labels along the self-motion path (-1, 1, -1) | travelled 12.00 rad, min clearance 8.87e-02, pose drift 3.1e-15 | labels constant
  solution 5: labels along the self-motion path (-1, 1, 1) -> (-1, 1, -1) | travelled 12.00 rad, min clearance 7.09e-02, pose drift 6.4e-13 | LABELS CHANGE

solutions whose labels change along a pose-invariant, clearance-keeping path: 4 of 6
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

## `python3 -m study.exp30_adaptive_scan --pose-seeds 0 --uniform-grid 5 --uniform-seeds 80`

```
pose seed 0: coarse counts [4, 4, 4, 4, 8] | adaptive 8 solutions in  13.7 s | uniform 8 solutions in  10.3 s | adaptive misses 0 of uniform, uniform misses 0 of adaptive
```
