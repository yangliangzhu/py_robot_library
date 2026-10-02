# 共读清单：分支结构、cuspidality、与 real monodromy

起因：本课题（六轴臂 IK 的"解分支"判定 + 延拓完备求解）在机器人学里**有名字、有历史、有活跃的近作**，
只是术语不同。这份清单把该读的东西按"能回答我们哪个问题"分组，并标注**要从里面取什么**。

对照表（我们的词 ↔ 文献的词）：

| 本课题用词 | 文献用词 | 含义 |
|---|---|---|
| chamber | **aspect** | 关节空间里不含奇异位形的极大连通区域（Borrel & Liégeois 1986） |
| 一个解 / 沿曲线的分支 | **posture** / uniqueness domain | 一个离散逆解；沿笛卡尔曲线连续提升出的那一条 |
| 我量到的"14 对不穿奇异即换解" | **cuspidality** | 该臂可非奇异换位形 |
| 覆盖层数 `d_C ≈ 6` | 一个 aspect 里的**唯一性域个数** | 由 **characteristic surfaces** 切开（Wenger 1992, 2004） |
| `sign det J` 证书 | 同一 aspect 的**必要条件** | `det J` 在 aspect 内不变号 |
| 直线路径穿 Σ **两次**仍同支 | 路径的**工作空间投影绕过尖点** | cusp 是 cuspidality 的成因（El Omri & Wenger 1995） |

---

## A. cuspidality / aspects（机器人学侧，和我们最贴近）

1. **P. Wenger, "Cuspidal Robots"（综述章节）** — <https://ar5iv.labs.arxiv.org/html/1610.04080>
   取：aspect 的定义；"cuspidal = 某个 aspect 里多于一个逆解"；**特征曲面**的定义
   `CS_i = f⁻¹(f(A_i*)) ∩ A_i`；唯一性域；cuspidal 臂"无法跟踪某些笛卡尔路径"。
   **这份直接回答了本课题第一题的"两定义是否等价"**：等价 ⟺ aspect 就是唯一性域。
2. **Borrel & Liégeois 1986, ICRA** — <https://www.semanticscholar.org/paper/f2adab4b16fe91cd466c8ce8f16103c51f05925e>
   取：aspect 概念的出处；以及"当年那个**错误的证明**"（断言换解必穿奇异）——历史教训值得引。
3. **El Omri & Wenger 1995 / Wenger 1992** — 经上面综述转引。
   取：**尖点（cusp）判据**：奇异轨迹里出现尖点 ⇒ 该臂 cuspidal（绕尖点走即可换解）。
   对我们：本课题量到"直线穿 Σ 两次仍同支"的 8 对，理论上应该是**投影绕过 cusp** 的路径 —— 值得直接验证。
4. **Wenger 2004, 最大唯一性域** — 转引自综述。
   取：把一个 aspect 切成唯一性域的**最大**做法；对应我们量到的 `d_C ≈ 6`。
5. **Wenger 等, "Kinematic issues in 6R cuspidal robots, guidelines for path planning and deciding
   cuspidality", IJRR 2024** — <https://dl.acm.org/doi/10.1177/02783649241293481>（HAL 版：
   <https://hal.science/hal-04712576v1>）
   取：**6R 上如何判定 cuspidality**、离线编程的路径规划指南。这是与我们第一题**正面重合**的最新工作，
   必须逐条对比：他们的判据 vs 我们的 `sign det J` 证书 + 见证路径 + 覆盖层数。
   ⚠️ 目前只读到题目与摘要级信息（PDF 抓取受限），**尚未逐条核对**。
6. **"Trajectory planning issues in cuspidal commercial robots"（2023）** —
   <https://hal.science/hal-04055520v1/document>
   取：**真实商用机型**上的轨迹规划问题与对策 —— 与提问者的 A→B 规划场景最接近。
7. **"Hidden cusps"（Wenger 等, 2016）** — <https://ar5iv.labs.arxiv.org/html/1604.08742>
   取：不是所有 cusp 都在明面上；判定不能只看显式奇异集。

## B. real monodromy / Galois（数学侧，第三话题的现代家园）

1. **Hauenstein, Regan, Sherman, Wampler, "Computing real monodromy"** —
   <https://academicweb.nd.edu/~cwample1/Preprints/hrswComputingRealMonodromy.pdf>
   取：核心事实 —— **复单值化群可以是对称群，而"实单值化"仍然分解**；算法：用实路径/实回路跟踪实解，
   得到实解集合上的置换群，从而**分解实解集**。这正是"哪些实解能连续互达"的可计算版本，
   与我们的 14 对见证路径是同一件事的两面。
2. **M. Regan, "Real Monodromy Action"** — <https://eventos.ucm.es/_files/_event/_24016/_editorFiles/file/MEGAR_poster_abstract_Regan.pdf>
   取：问题的现代表述与动机（"为什么复的答案不够、实的世界有更多结构"）。
3. **Duff, Leykin, Pajdla 等, "Galois/monodromy groups for decomposing minimal problems in 3D
   reconstruction"** — <https://ar5iv.labs.arxiv.org/html/2105.04460>
   取：在视觉领域已经落地的做法：算单值化群 → 把多项式系统分解成更小的块。
   对我们的启示：同一套工具可以用来问"6R IK 的单值化群是什么、可解否"。
4. **RealNAG（real numerical algebraic geometry）2025 阅读清单** —
   <https://qubeshub.org/community/groups/realnag2025/reading_list__research_problems>
   取：这一小块领域当前的**社区与开放问题**（判断它是否真"冷门"的最好证据）。

## C. 延拓 / 多项式系统（方法基底）

1. **Tsai & Morgan 1985**, "Solving the kinematics of the most general six- and five-degree-of-freedom
   manipulators by continuation methods", ASME J. Mech. Transm. Autom. Design 107:189–200.
   取：**我们这条路的原始出处**；"一般 6R 不超过 16 个解"；延拓的两阶段（起始系统 → 同伦路径 → 牛顿）。
   Angeles 教材 §9 对此有中文概述（`/home/yang/translation/processing_workflow/angeles/ch09a.typ`）。
2. **Morgan 1987**, *Solving Polynomial Systems Using Continuation for Engineering and Scientific
   Problems*（教材级）。
3. **Sommese & Wampler**, *The Numerical Solution of Systems of Polynomials Arising in Engineering
   and Science*（数值代数几何的权威教材；IK 是书中的范例之一）。
4. 软件：**Bertini**、**HomotopyContinuation.jl** —— 若要把"用单值化群判可解性"做成工具，这两个是现成基座。

## D. 我们要从文献里取的答案（逐条）

| 我们的问题 | 该去文献里找什么 | 现状 |
|---|---|---|
| 分支划分是否等于 aspect？ | Wenger 综述 §6/§7 + 特征曲面定义 | **已有答案**：cuspidal 臂上不等价，差一个"唯一性域"细分 |
| 6R 上如何判定 cuspidality？ | IJRR 2024（第 5 条） | **未读**，需逐条对比我们的证书 |
| 直线穿两次仍同支的机制？ | cusp 判据（El Omri & Wenger 1995） | **假设成立待验证**：验证我们那 8 对的投影是否绕过 cusp |
| 6R IK 能否用根式解？ | real monodromy 文献（B 组）+ Pieper/Mavroidis-Roth 的可解结构分类 | 机器人学侧**未见**；数学侧工具有（B 组） |
| 冗余臂上"分支"该怎么说？ | 本课题已测（Burdick 1988 的 self-motion manifold 概念是早期工作） | 我们的"分量下界 13"是工程性测量 |

## E. 待办（可选，按价值排序）

1. **读透 IJRR 2024**（第 5 条）：把他们的 cuspidality 判据与我们的结果逐条对齐，明确哪些是我们的重复、
   哪些是补充。若拿不到正文，先找可访问的预印本/HAL 版。
2. **验证 cusp 机制**：取我们那 8 对"直线穿 Σ 两次"的见证路径，把关节空间路径投影到工作空间，
   检查投影**绕过奇异像的尖点**；再做一条故意不绕尖点的路径作对照（预期走不通）。
3. **算 SR5 的特征曲面 / 唯一性域**：用现有工具（增广系统求 fold、σ_min 见证路径）判定"位姿是否落在
   奇异像上"，把一个 aspect 切成 6 个唯一性域 ⇒ 同时完成"完整分支划分"与"可行路径域"。
4. **real monodromy 探针**：按 B 组算法跟踪实解回路，算 SR5 的实单值化置换群；用 ROKAE ER3（真 S-R-S）
   与 franka（非球腕但可解）作对照，看实单值化结构是否区分"可解/不可解"。

## F. 一句提醒

本课题里凡是"看起来没人做过"的想法，先按 A/B/C 三组各搜一遍再下结论：本次核对的结果是
**现象与几何定义（aspect/cuspidality）已有 40 年历史且 2024 年仍在 IJRR 上活跃**，而
**"实单值化群 = 分支结构"这一层主要在数学/视觉圈，机器人学侧少见**。
提问者是独立走到同一处的（未读过这些），这说明该框架是问题**逼出来**的，而不是某个人的偏好。
