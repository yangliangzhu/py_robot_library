# IK branch structure and DH continuation — research log

Branch `study/ik-branch-and-continuation`. Everything here is exploration: `study/` is
deliberately outside `pyproject.toml`'s `testpaths`, so a half-finished experiment cannot turn
the library's suite red. Library code is never modified from here — findings about it are
written down instead (see §4).

Two questions, both from L. Yang's NetEase work on the ROKAE SR4 (a 6R whose wrist axes do not
intersect, so the engineers' "no closed form" is right and Pieper does not apply):

1. **Branch membership.** Given two solutions `q1 != q2` of one pose `T`, are they in the same
   IK branch? The empirical criterion was: interpolate a path in joint space; if singularities
   are the boundaries between branches, the path must cross the singular set, which shows up as
   a manipulability dip. Experimentally strong on SR4, with counterexamples where the dip was
   sharp but never reached zero.
2. **DH continuation.** With `FK(dh, q) = T`, differentiating gives `J_dh dh + J_q dq = 0`, so a
   solution can be *continued* as the DH parameters vary. For SR4/SR5, a few parameters can be
   perturbed to reach a Pieper-solvable arm ("SR0"), so: solve SR0 in closed form, then continue
   to the real arm.

## 1. Vocabulary, and why the two questions are one

* `Σ` — the **singular set** `{q : rank J(q) < 6}`. Measured by `sigma_min` (smallest singular
  value), which is zero exactly on `Σ` and non-negative everywhere. The library's manipulability
  is a *product* of singular values: it is the same zero set, but it decays as a high power, so
  it is a worse instrument for deciding "did we cross?" — see §4.3.
* **chamber** — a connected component of `Q \ Σ`, `Q` being the configuration space: a product
  of circles (continuous joints) and intervals (limited joints).
* **fiber** over `T` — `{q : FK(q) = T}`. Finite for a 6R; a union of curves for a redundant arm.
* **branch** — a connected component of `fiber \ Σ`: one solution, or one arc of a self-motion
  curve between the singular configurations on it.

Three statements, in decreasing strength:

* **(exact, and a tautology)** two configurations in different *chambers* cannot be joined
  without crossing `Σ`. This is the direction that yields certificates.
* **(the real claim)** two distinct solutions of the same regular `T` are in different chambers.
  If it holds for an arm, "same branch" and "same chamber" coincide and the empirical criterion
  is exact in principle.
* **(the empirical criterion)** the crossing can be *detected* as a manipulability dip along a
  path. This is where counterexamples live: a dip is not a crossing, and a fine enough
  interpolation of a crossing may not dip at all.

The useful reframing: the criterion is a *test* (does this path cross `Σ`?) for a *definition*
(are these in the same chamber?). Testing it needs a crossing detector, and the robust detector
is not the dip but the refined minimum of `sigma_min` along the path — because "crossing" means
`sigma_min = 0` somewhere, and a golden-section refinement separates an exact zero (crossing)
from a small positive value (near miss). Finding 2 then closes the loop: homotopy continuation
is both the thing being studied and the only *complete* census tool for the fibers Finding 1
needs.

**Prediction to test**: the criterion can only be exact for non-redundant arms. On a 7-DOF arm
all solutions of one pose lie on one self-motion curve, so they are generally connected without
crossing `Σ`; the branch structure there is the partition of that curve by the singular
configurations on it, and the criterion must be applied *along the curve*, not along an
arbitrary joint-space path. The ER3 measurements in §3.3 already show the continuum.

## 2. Code

| file | what it does |
|---|---|
| `study/ik_structure.py` | `sigma_min`, configuration-space distance/paths (circles vs intervals), `find_fiber` (multi-start + dedupe), `scan_path` (coarse bracketing + golden-section refinement of every local minimum of `sigma_min`) |
| `study/census.py` | `solve_lm` (census-grade Levenberg-Marquardt, `1e-9`), `census`, `evaluate_solvers`, `perturbation_sweep` |
| `study/exp00_solver_baselines.py` | solver baselines, coverage curves, the solution count |
| `study/exp01_ik_standard_failure.py` | what `IkStandard`'s `ok=False` means |

## 3. Measured so far

### 3.1 A 6R with an offset wrist has few real solutions, and a census is not cheap

`python3 -m study.exp00_solver_baselines --robot sr5 --backend casadi --seeds 400`

* Solutions found at random regular poses: **8, 8, 8** in one run, **4, 4, 5** in another
  (different poses). The count varies with the pose, so "up to 8" is a property of the arm and
  the pose, not a constant.
* Basin sizes are nearly uniform when the count is small: at one pose, 8 solutions from 551
  converged seeds had hit counts `[99, 93, 72, 66, 64, 57, 53, 47]`.
* Rare solutions exist: at one pose a 5th solution appeared once in 800 seeds (hit count 1 of
  222 converged). Any census that stops when it stops finding new solutions can therefore miss
  a solution with a small basin — which is exactly why §5.3 wants a homotopy census.
* All solutions found were regular: `sigma_min` between 0.01 and 0.21.

### 3.2 Solver baselines (SR5, casadi, 150–400 seeds)

| solver | from random seeds | from a 0.05 rad perturbation | cost |
|---|---|---|---|
| `IkStandard` (library default) | 0.0–0.2% | 92–94% to *some* solution, 92–94% to the *same* one | 3.0 ms/attempt |
| `solve_lm` (study) | 26–62% | 100% | 7.6 ms/attempt |

* Both backends agree **bit for bit** on all of this (same rates, same solutions, same
  `sigma_min`); casadi is ~1.6x faster per attempt.
* The library solver is *not* jumping branches under perturbation — an earlier reading of mine
  that suggested it was, was an artifact of re-checking its output against `1e-6` while its own
  convergence tolerance is `1e-5`. Corrected in §3.4.
* A 6R's branch structure is not reachable by random restarts alone: `IkStandard` finds a
  solution from 1 seed in 500; `solve_lm` from 1 in 2. The census uses `solve_lm`.

### 3.3 A 7-DOF arm's fiber is a continuum (prediction confirmed early)

`python3 -m study.exp00_solver_baselines --robot er3 --backend casadi --seeds 800`

* 370 "distinct solutions" at a `1e-4` dedupe tolerance, all with `sigma_min ~ 0.17`: these are
  points along one self-motion curve, not branches.
* From a 0.05 rad perturbation of a known solution the library solver reached *some* solution
  91% of the time but *that* solution only 4% — on a redundant arm even a tiny perturbation
  slides along the manifold, so "same solution" is the wrong invariant there.
* Consequence for Finding 1: for a redundant arm the branch structure must be computed on the
  self-motion manifold (arcs cut out by `Σ`), and the empirical criterion is then a statement
  about *that* curve. This is testable and is the first experiment of round 2.

### 3.4 Two defects in `IkStandard`'s failure path (side finding, measured)

`python3 -m study.exp01_ik_standard_failure --samples 300` (SR5, casadi, 0.2 rad perturbations;
the tracing subclass is verified against the library class on 20/20 seeds first):

* 72/300 seeds fail. Of those failures, the best pose error the solver had *already reached* was
  a median of `5.1e-3` (90th percentile `3.6e-1`) — most failures are genuine, not near-misses.
* **The returned configuration is not the best one visited.** Worst case: the returned pose
  error is `1.51e-1` worse than an iterate the solver had already been at. Cause: both failure
  returns (`step_size < 1e-4` and the loop exiting) return the current `angle`, while `q_best` is
  only restored on the branch that survives the step-size test (`model/ik_solver.py:147-171`).
* **15% of failures were already within `1e-4` of the target** — i.e. `ok=False` sometimes means
  "stopped at the accuracy floor", because the step-size back-off is monotone non-increasing and
  near `Σ` (median `sigma_min` at the best iterate on failures: `0.019`, half the failures below
  `0.02`) the damped-least-squares step oscillates.
* Both matter beyond this study: a control loop that gates on `ok` rejects usable solutions
  near singularities, and any census that trusts `ok` under-counts solutions.

## 4. Next

1. **exp02 — chamber census on SR5.** For each pose: all solutions (multi-start, `solve_lm`),
   then every pair scanned with the refined-minimum test. Report: how often a pair's shortest
   path crosses `Σ` exactly; the distribution of the refined minimum (crossing vs near miss);
   the same pairs under the manipulability-dip criterion at several interpolation densities, to
   show where the empirical criterion fails and by how much.
2. **exp03 — analytic ground truth.** A 2R and a 3R planar arm (closed-form branches) and the
   library's own `model/ik_srs.py` (S-R-S 7-DOF, closed-form branches) as ground truth: measure
   false-positive/false-negative rates of the dip criterion, and of the `det J` sign certificate
   (`sign det J(q1) != sign det J(q2)` proves "different chamber" in one Jacobian each).
3. **exp04 — the trajectory experiment, reproduced.** Multi-start census + continuity-based
   labeling along a trajectory, exactly as in the original experiment, then the same labeled
   pairs judged by (a) the dip criterion, (b) the refined-minimum crossing test, (c) the sign
   certificate. The point is to explain the counterexamples rather than to re-confirm the rule.
4. **exp05 — Finding 2.** `J_dh`/`J_q` first-order continuation: error versus step size, failure
   near `Σ`; then a predictor-corrector tracker in `(dh, q)`; then the SR0 construction (which
   parameters to relax, decided from the joint-axis geometry rather than by trial), an analytic
   SR0 solver, and continuation SR0 -> SR5 including branch switching at the discriminant.
5. **exp06 — homotopy as a complete census**, i.e. the tool §3.1 says is missing: track the
   solutions found at one pose around a loop in the workspace and see whether new ones appear.
