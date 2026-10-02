# Day 3 实验结果（kimi 侧）

> 协议第 1 条：命令 + 数字。写作者：Kimi Code CLI，2026-10-03，基线提交 `8face7f`。
> 原始日志：`/tmp/study_verify/mono_tally_{sr5,sr0}.log`、`exp34_s{1,2}.log`。

## R7. SR5 22 回路（复跑 `8face7f`，逐行一致）

命令：`python3 -m study.monodromy --seeds 600`

* 自测：`certificate self-test passed: D5 rejected, A5 certified`。
* 全员存活 5 条（`position xy/xz r=0.002`、`orientation z/x a=0.02`、`orientation x a=0.05`），
  映射全部恒等；15 条有死亡，**存活者移位计数全为 0**。
* `tally: 22 loops | all-survivors 5 (non-identity among them: 0) | some sheets died 15 |
  prediction 'no deaths => identity': HELD`
* 失败分类样本：`met the singular set` 出现于 `position xy r=0.02`（解 3、6，间隙 1.7~1.8e-3）；
  `left the joint limits` 多条；两条整圈回路 0/8（越限 5 + 丢失位姿 3 等混合）。

## R8. SR0 控制组 22 回路（kimi 独立复跑，与 f544050 的 14 回路版本参数不同）

命令：`python3 -m study.monodromy --robot sr0 --seeds 600`

* 基位姿 8 解（环面去重后；与"15 代表值 = 8 环面类"一致）。
* **全员存活 7 条**：`position xy/xz r=0.002`、`r=0.005`、`orientation z/x a=0.02`、
  `orientation x a=0.05` —— 全部恒等。预言在控制臂上非空检验通过。
* **`orientation full turn z`：4/8 存活且恒等**（map `{0:0, 2:2, 3:3, 5:5}`，
  min clearance 4.56e-3）；失败者 `1/4/6/7` 全部 `left the joint limits`，无一撞奇异。
  ⇒ 整圈回路可跟踪，障碍是限位不是奇异；SR5 的 0/8 是该臂事实，非拓扑必然。
* `orientation x` 各档（a=0.2/0.4/0.8 与整圈 x）：**0/8，全部 `met the singular set`
  (1.2~1.5e-3)** —— 球腕纤维必经腕奇异层的干净签名。
* 失败分类对照（与 SR5 相比）：SR0 的失败以"越限"与"撞奇异"为主，
  `lost the pose` 远少于 SR5 —— 与"SR5 失败多为几何（fold 多）"的方向一致。
* `tally: 22 loops | all-survivors 7 (non-identity among them: 0) | some sheets died 11 |
  prediction ...: HELD`

## R9. 残余猜想：两个新残余均闭合（`study/exp34_residuals.py`）

命令：`python3 -m study.exp34_residuals --seed 1 --poses 3`、`--seed 2 --poses 1`
（replay exp19 的 rng 流，逐 pose 复算划分；孤立解审讯同 exp32）

**seed 1 pose 2**（`[2,2,1]`，5 解，2+/3-，孤立解 1 属 −1 类）：

```
singleton 1: sigma_min 8.995e-02, nearest solution 3.214 rad
pair (1,3): line crosses det J 4x | long walk connected=False (70%, budget) | retries [F,F,F]
pair (1,4): line crosses det J 2x | long walk connected=False (61%, budget) | retries [T,T,T]
```

⇒ `1~4` 连通 ⇒ 闭合为 **[2,3] = 符号划分**。（`(1,3)` 直线穿 Σ 四次仍应同支——绕路×2。）

**seed 2 pose 0**（`[2,1,1]`，4 解，2+/2-，孤立解 0、2 均属 −1 类）：

```
singleton 0: sigma_min 1.561e-01, nearest 4.557 rad
pair (0,2): line crosses det J 2x | long walk (64%, budget) | retries [T,T,F]
singleton 2: sigma_min 1.608e-01, nearest 4.661 rad
pair (2,0): line crosses det J 2x | long walk (99%, budget) | retries [T,T,T]
```

⇒ `0~2` 连通 ⇒ 闭合为 **[2,2] = 符号划分**。

**3/3 残余（含 exp32 的 `[4,3,1]`）全部为走法预算 artifact；划分 = `det J` 符号划分在
全部受测位姿上成立。**
