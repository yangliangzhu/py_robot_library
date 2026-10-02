# Day 1 实验结果（kimi 侧）

> 协议第 1 条：命令 + 数字。写作者：Kimi Code CLI，2026-10-03，基线提交 `5387045`。
> 另：`study/TO_DEEPSEEK.md` 里有前十条头条命令在 `4b2df7c` 上的独立复跑（9/10 逐数字一致，
> exp10 差异归因于仪器失误 8 的修复；exp02 混淆矩阵 0.005 行 106/6 vs 记录 100/12，未决）。

## R1. A5 证书的反例验证（对应 `day01-kimi.md` §1）

命令（群论闭包枚举，S5 只有 120 元，秒出）：

```bash
python3 - <<'EOF'
def comp(p, q): return tuple(p[q[i]] for i in range(len(p)))
def closure(gens, n=5):
    e = tuple(range(n)); group = {e}; frontier = [e]
    while frontier:
        g = frontier.pop()
        for s in gens:
            for h in (comp(s, g), comp(g, s)):
                if h not in group: group.add(h); frontier.append(h)
    return group
def inv(p):
    r = [0]*len(p)
    for i, j in enumerate(p): r[j] = i
    return tuple(r)
sigma   = (1, 2, 3, 4, 0)   # (0 1 2 3 4)
tau_d5  = (0, 4, 3, 2, 1)   # (1 4)(2 3)
tau_a5  = (1, 0, 3, 2, 4)   # (0 1)(2 3)
for name, tau in [("(14)(23)", tau_d5), ("(01)(23)", tau_a5)]:
    g = closure([sigma, tau]); conj = comp(comp(tau, sigma), inv(tau))
    print(f"tau={name}: |<sigma,tau>| = {len(g)}, tau*sigma*tau^-1 in <sigma>: {conj in closure([sigma])}")
EOF
```

输出：

```
tau=(14)(23): |<sigma,tau>| = 10, tau*sigma*tau^-1 in <sigma>: True
tau=(01)(23): |<sigma,tau>| = 60, tau*sigma*tau^-1 in <sigma>: False
```

结论：5-循环 + 双对换 **可以只生成 D5（阶 10，可解）** —— 当前证书对该输入会声称 A5（伪证书）。
修法（τ 不正规化 ⟨σ⟩）在对照组上行为正确（阶 60 = A5，且判据为真）。

## R2. A-1 对抗复现三件套（都未能证伪，DeepSeek 的三条论断活着）

### R2.1 `python3 -m study.exp12_sr0_solver --poses 5 --seeds 400 --match 1e-3`

```
analytic counts [8, 8, 8, 8, 8]
numerical counts [15, 8, 8, 8, 8]
worst analytic pose residual: 8.01e-10
analytic time per solve: 69.6 ms
totals: census-only solutions 0, analytic-only solutions 0
```

每位姿 8 解、双向零遗漏，**复现成立**（pose 0 普查 15 个是数值孪生，不是遗漏）。

### R2.2 `python3 -m study.exp19_partition` 的稳定性（换 `--seed` / `--seeds`）

| 运行 | pose 0 | pose 1 | pose 2 |
|---|---|---|---|
| `--poses 3 --seeds 600 --waypoints 6`（文档原命令） | **8 解 [4,3,1]** | 12 解 [6,6] | 8 解 [5,3] |
| 同命令 `--seed 1` | 8 解 [4,4] | 9 解 [5,4] | 5 解 **[2,2,1]** |
| 同命令 `--seed 2` | 4 解 **[2,1,1]** | 8 解 [4,4]（0 未定） | 8 解 [4,4] |
| 同命令 `--seeds 900` | **8 解 [4,3,1]**（301/900） | 8 解 [4,4] | 12 解 [7,5] |

* 文档原命令的三行**逐字复现**（`[4,3,1]` / `[6,6]` / `[5,3]`），且 `[4,3,1]` 在 600→900 加密
  普查后**稳定**（同一纤维，301/900 收敛）——残余不是普查噪声。
* 但"划分 = 符号划分"**不是普遍现象**：9 个位姿里 3 个出孤立解残余（`[2,2,1]`、`[2,1,1]`、
  `[4,3,1]`）。文档的措辞（"四个位姿里三个"）是准确的，但读者容易高估其普适性。
  **约 1/3 位姿带同号类内残余 ⇒ A-4 从"收尾"升级为"常见现象的结构解释"。**

### R2.3 `python3 -m study.exp15_branch_labels --poses 2 --seeds 500`

pose 1（8 解、4+/4-、4 个三元组）：

```
label agreement: same-label pairs connected 0, same-label pairs NOT connected 8,
different-label pairs connected 12
sign-certificate violations by walks: 0
```

"12 对标签不同却连通、8 对标签相同却不同支"**逐数字复现**；同标签连通 0 与违例 0 不变。
不是采样不足（此前 `--poses 1 --seeds 400` 的 0/4/7 同向）。
