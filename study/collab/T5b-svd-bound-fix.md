# T5b — SVD 下界公式与内积长度修正

2026-10-17。按 [GPT_NEXT_GATE_20261017.md](GPT_NEXT_GATE_20261017.md) §2 的可转交任务 T5b，
修正 `study/exp63_segment_certificate.py` 中三处数学/舍入缺陷，补回归，重跑原 SR5 路径对账。

## 1. 修正前/后的反例

GPT 的精确反例：1×1 矩阵 `U = A = [sqrt(0.9)]`，`s = V = 1`，故 `delta_U = 1/10`、
`delta_V = delta_R = 0`，真实奇异值 `sqrt(0.9) ≈ 0.9486833`。

| 版本 | `_svd_sigma_min_lower` 返回值 | 是否合法下界（≤ 真值） |
|---|---|---|
| 旧公式 `1 - (δU+δV)/2` | `0.9499999999999986` | **否**（`0.95 > 0.94868`） |
| 新公式 `sqrt((1-δU)(1-δV))` | `0.9486832980505124` | 是（等于真值到 1e-15） |

复现命令（修复前会打印 `true: 0.9486832980505138 reported lower: 0.9499999999999986`）：

```bash
python3 - <<'PY'
import numpy as np
from study.exp63_segment_certificate import _svd_sigma_min_lower
a = np.array([[np.sqrt(.9)]])
bound = _svd_sigma_min_lower(a, a, np.array([1.]), np.array([[1.]]))
print('true:', a[0,0], 'reported lower:', bound)
PY
```

## 2. 三处修正

1. **下界公式**（`_svd_sigma_min_lower`）：`s_min * (1 - (δU+δV)/2) - δR` → `s_min * sqrt((1-δU)(1-δV)) - δR`，
   并对整体结果截到 `>= 0`。`1 - (δU+δV)/2` 是 `sqrt((1-δU)(1-δV))` 的**上界**（算术–几何平均），
   不能用作下界；`sqrt((1-δU)(1-δV)) = σ_min(U)·σ_min(V)` 的下界因子才是对的。

2. **`UᵀU` 内积长度**（`_svd_sigma_min_lower` 的 `delta_u`）：舍入膨胀用 `_gamma(singular.size)` 改为
   `_gamma(left.shape[0])`。`UᵀU` 每项是 `U` 两列（长 `U.shape[0]`）的点积；高（tall）矩阵时
   `U.shape[0] > singular.size`。

3. **Gram 舍入界**（`_inverse_sigma_min_lower`）：`_gamma(min(rows, columns))` 改为
   `_gamma(max(rows, columns))`。宽 `A` 的 `AAᵀ`（或高 `A` 的 `AᵀA`）每项内积长度是**较长**的一维。

`machine_verified=False` 保持不变（分析界 + 浮点误差估计，区间后端仍是第二阶段，未升级标签）。

## 3. 新增回归（`phase_svd_regressions`，运行于 `main()` 的 phase 0）

- 1×1 反例：断言返回 ≤ `sqrt(0.9)` 且 ≈ `sqrt(0.9)`（旧代码给 0.95，会失败）。
- 高 (5×2) SVD 因子反例：`U = diag(sqrt(0.9), 1)` 补零，断言 ≤ `sqrt(0.9)`（旧代码 0.95，失败）。
- 非方阵（tall 7×3 / wide 3×7）经 `sigma_min_sample_bounds` 的合法性检查。
- 高 `UᵀU` 长度（issue 2）：恰正交的 (100000×2) `U` 使 `|UᵀU-I|=0`，`delta_u` 退化为 `gamma(rows)`；
  断言返回值 ≤ 用 `gamma(rows)` 手算的严格界（旧代码用 `gamma(2)`，会高估而失败）。
- Gram 长度（issue 3）：全 1 的 (1×N) 行与 (N×1) 列，`AAᵀ = N` 精确可算，断言逆路线返回值 ≤
  用 `gamma(N)` 手算的严格界（旧代码用 `gamma(1)`，会高估而失败）。

## 4. 原 SR5 路径重跑对账

`python3 -m study.exp63_segment_certificate`（full run，`--pose-seed 0`，即 exp49 seed 0 的对 (1,6)）：

| 量 | 旧（记录值） | 新（本次重跑） |
|---|---|---|
| 认证下界 `certified_lower` | `4.9829e-3`（打印到 5 位有效数字） | `4.9828928e-3`（= `0.004982892796488046`） |
| 采样最小值（旧仪器） | `5.245e-3` | `5.245129e-3` |
| 认证段数 | 118/118 | 118/118（仍 `CERTIFIED`） |
| `delta_ok`（> 5e-3） | False | False |

新值 `4.9828928e-3` 四舍五入到 5 位有效数字仍是 `4.9829e-3`，与记录值逐字一致；两者差的绝对值
`7.2e-9`（≈ 0.00014% 相对）来自旧打印值的舍入，远小于括号容差（`rel_tol=0.05`）与采样/认证的任何
有意义量级。

**结论变化：无。** 该路径"同 aspect（同一分支）"的认证结论不变：`sigma_min > 0` 整段证书仍成立
（118/118 段），认证下界仍为 `4.9829e-3`，`5e-3` 这一整数门槛仍未被认证（`4.98e-3 < 5e-3`）。
这与 GPT §2 的预期一致：对正常 SVD，δU/δV ~ 1e-15，错误公式与正确公式的差是二阶项（~1e-30），
远小于 Lipschitz 余量，故否定的是"严格下界"的**一般证明**，而不是这条 SR5 路径确实无奇异。

## 5. 复现状态

- `python3 -m ruff check .` → clean（exit 0）。
- `python3 -m pytest -q` → `217 passed`。
- `python3 -m study.exp63_segment_certificate` → `34/34 checks passed`，exit 0。

未提交（按任务要求不 commit）。未决/边界：非方阵的两处**长度**缺陷是 epsilon 级严格性修正
（正常输入下数值上仍合法），回归用"精确正交/全 1 + 大内积长度"的构造来让旧代码的**高估**可见；
没有一种值级构造能让旧代码在 `sigma_min_sample_bounds` 的普通随机非方阵上越界。
