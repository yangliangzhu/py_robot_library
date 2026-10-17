# T4 — Gate C2：验证器完整性修复 + 两个连续参数段的精确端点公式证书（exp71）

2026-10-17，DeepSeek 子代理 `gate-c1-roots`。任务 = GPT 指派：①先修 C1 验证器完整性缺陷（`GPT_C1_REVIEW_C2_ADDENDUM_20261017.md` §"C1 verifier completeness defect"）；②C2 原型（同文档 §"C2: exact endpoint formula" 与 §"C2 deliverable and acceptance"）。**只做 C2**：无三环认证、无置换声明、无 `path_certified` 类标志。

## 1. 第一步：C1 验证器完整性修复（exp70 模块内，证书数字不变）

**缺陷复现（修复前，与 GPT 报告一致）**：把证书 JSON 的 `roots` 清空后 `--verify` 返回 **exit 0 且打印 PASS**；把根 6 标成 `inconclusive` 同样 **exit 0 PASS**。

**修复内容**（`study/exp70_gate_c1_roots.py` 验证器重构为 `verify_payload`）：

1. **完整 C1 验收**要求 `roots` 键集合**恰好**为 {1,4,6,8,10,13}（缺根/多根/空集合全部非零退出），每个根 `verdict` 必须为 `certified`（未知 verdict、`inconclusive` 一律拒绝）；
2. **结构校验**：centre/centre0/seed_q 12 维、B 12×12、radius_k 非负整数、全部数值字段必须可解析为有理数，畸形记录显式 FAIL；
3. 保留全部六个收缩与 15 对分离检查的精确重算，并**新增** C1 选盒策略复检（κ < 1/2 且 margin ≥ r/2，按重算值）；
4. 单根复核是**独立模式**（`--verify-roots 1`），输出明示 `single-root mode; NOT full C1 acceptance`，不可能与完整 C1 验收混淆；
5. **负对照回归**（每次 `--verify` 自动先跑，7 项全部按预期拒绝）：空 roots、缺根 6、根 6 标 inconclusive、未知 verdict、centre 维度 11、B 维度 12×11、多余根键。

**修复后证据**：
- 空 roots → exit 1（`roots keys exactly [...] (got [])`）；inconclusive → exit 1；单根文件按单根模式 exit 0、按完整 C1 exit 1；
- 未改动的原证书 `study/exp70_gate_c1_certificate.json`：`python3 -m study.exp70_gate_c1_roots --verify ...` → **exit 0**，`PASS: certificate fully re-checked (full C1: 6/6 roots {1,4,6,8,10,13} certified)`。

**六盒本身不受影响**：本步只改验证器的完整性判定，未重跑认证、未改动任何证书数字；原证书 JSON 逐字节未变，修复后的验证器原样重新接受它。

## 2. 第二步：C2 原型——精确端点公式的两个连续段

**段几何**：exp68 主回路首腿从 `u=0` 沿**虚轴直线**上行（`u=i·t`，位姿 `p0+u·d`，`d=(-1,0,0)` 冻结自归档）。认证段取 `λ=i·s`，s 为二进有理：段 1 = `[0, i·2⁻²⁵]`，段 2 = `[i·2⁻²⁵, i·2⁻²⁴]`，两者都落在首个归档节点区间 `[0, i·1.21e-3]` 内（同一直线，关系可核对；`2⁻²⁴ ≤ Im u₁` 已精确验证）。

**端点公式**（addendum）：固定姿态、线性平移 ⇒ `f(x,λ)=f_base(x)−g(λ)` 对 λ 仿射、`Df` 与参数无关；`v(λ)=Bf(c,λ)` 满足 `v(λ(t))=(1−t)v_A+t·v_B` **精确成立**——每条管都存了中点的**精确等式检查**（`2v_mid == v_A+v_B`，非采样假设）；故 `η_i = max(|v_A[i]|,|v_B[i]|)` 是整段逐行残差的**精确界**。配合整盒区间界 `κ_i ≥ sup_X‖row_i(I−BDf)‖₁`（exp70 的精确有理区间后端，无浮点舍入），逐行严格余量 `r−η_i−κ_i·r>0` 加 `κ<1` ⇒ 对每个 t，`T_t(x)=x−Bf(x,λ(t))` 是管盒到内部的严格收缩 ⇒ 整段唯一正则根、随 t 连续。假根排除（`|trace S−3|<4`、`z_j≠0`）与 C1 相同。选盒策略：κ<1/2 且 margin ≥ r/4，梯子取最小可行 `r=2⁻ᵏ`；失败只记 未决，不放宽。

## 3. 逐根结果（全部 certified，证书 `study/exp71_gate_c2_certificate.json`）

| 根 | 段 1 管 | 段 2 管 | 拼接 E⊂int(X₂) 最小余量 | C1 盒⊂int(X₁) 最小余量 | 交换负对照 |
|---|---|---|---|---|---|
| 1 | r=2⁻²²=2.38e-7, η 1.41e-7, κ 8.70e-4, **41.0%** | r=2⁻²¹=4.77e-7, η 2.81e-7, κ 1.74e-3, **40.9%** | 4.77e-7 | 9.79e-8 | 换根 4 管：**拒**（−1.92） |
| 4 | r=2⁻²⁴=5.96e-8, η 3.22e-8, κ 7.82e-5, **46.0%** | r=2⁻²³=1.19e-7, η 6.44e-8, κ 1.56e-4, **46.0%** | 1.19e-7 | 2.74e-8 | 换根 6 管：**拒**（−1.73） |
| 6 | r=2⁻²⁴=5.96e-8, η 2.95e-8, κ 1.80e-4, **50.4%** | r=2⁻²³=1.19e-7, η 5.91e-8, κ 3.61e-4, **50.4%** | 1.19e-7 | 3.01e-8 | 换根 8 管：**拒**（−1.95） |
| 8 | r=2⁻²⁰=9.54e-7, η 4.83e-7, κ 2.10e-2, **47.3%** | r=2⁻¹⁹=1.91e-6, η 9.66e-7, κ 4.20e-2, **45.2%** | 1.89e-6 | 4.71e-7 | 换根 10 管：**拒**（−1.85） |
| 10 | r=2⁻²²=2.38e-7, η 1.21e-7, κ 6.10e-3, **48.6%** | r=2⁻²¹=4.77e-7, η 2.42e-7, κ 1.22e-2, **48.0%** | 4.75e-7 | 1.17e-7 | 换根 13 管：**拒**（−2.96） |
| 13 | r=2⁻¹⁹=1.91e-6, η 1.13e-6, κ 1.10e-1, **29.8%** | r=2⁻¹⁷=7.63e-6, η 2.26e-6, κ 4.40e-1, **26.4%** | 7.42e-6 | 7.78e-7 | 换根 1 管：**拒**（−2.31） |

（百分比 = 最小逐行余量 `r−η_i−κ_i·r` 占 r 的比例；trace 界 ≤5.1e-3、min‖z_j‖² ≥0.276、`‖I−BDf(c)‖∞` ≤8e-14、`det B≠0`，全部精确存证。中心/B 为冻结二进制有理数，管半径 2⁻ᵏ。）

**拼接链**：段 1 中心取 `exp(iq_base)` 与 λ_A 处浮点 Newton 根的冻结中点；`E = c₁−B₁f(c₁,λ_A)+(I−B₁Df(X₁))(X₁−c₁)` 精确区间含段 1 右端根；`E ⊂ int(X₂)` 逐坐标严格余量 + 两段唯一性 ⇒ 根身份链；`X_C1 ⊂ int(X₁)` 把链条锚到 exp70 六根。**负对照**：对每个根，把第二段换成另一根的有效管（单独证书全过），拼接检查全部**拒绝**（最小余量 −1.73…−2.96，远超盒尺度）——交换的不是标签而是真实证明检查。另有六个篡改负对照（空 roots、缺根 13、拼接端点错配、含 z_j=0 的管、篡改 direction、篡改模型 sha256）每次 `--verify` 自动执行、全部按预期拒绝。

## 4. 复核命令

```bash
# 完整验收（先重跑修复后的 C1 完整检查，再精确重算 C2 全部数字；exit 0）
python3 -m study.exp71_gate_c2_segment --verify study/exp71_gate_c2_certificate.json
# C1 验证器回归（7 个负对照自动先跑）
python3 -m study.exp70_gate_c1_roots --verify study/exp70_gate_c1_certificate.json
```

实测：C2 verify **309 PASS / 0 FAIL，exit 0**（含 C1 前置 `full C1: 6/6 roots certified`、6 篡改负对照、6 交换负对照）；验证器从冻结输入（exp69 模型 JSON、exp68 npz、exp70 证书、存证中心/B）重算每个有理数并精确比对，不信任 JSON 里的任何 PASS 字段。

## 5. 边界与质量门

- 只做 C2：两个连续段、六根各 2 条管、拼接与负对照；**不**做三环认证、**不**声明置换、**不**置 `path_certified`（JSON 中该字面量不存在）。
- 六根到真实单值化的桥接与不可解性措辞仍属 GPT（C3/C4 之后）。
- `ruff check .` 干净；`python3 -m pytest -q` = 217 passed；未提交。
- 改动文件：`study/exp70_gate_c1_roots.py`（仅验证器 + `interval_df_matrix` 共享重构，证书数字未动）、`study/exp71_gate_c2_segment.py`（新）、`study/exp71_gate_c2_certificate.json`（新）、`study/collab/QUESTIONS.md`（exp71 登记）、`study/collab/T4-gate-c2.md`（本注）。
