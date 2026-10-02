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

### 3.1 How many solutions a 6R with an offset wrist has, and why a census is not cheap

`python3 -m study.exp02_pair_census` (SR5, casadi, 400–600 seeds per pose)

* Counts found: **8, 8, 8, 8, 4** over five poses, and **8, 8, 9, 12, 6, 2** over another six.
  The count depends on the pose, and it reaches **12** — so "at most 8 solutions", the number the
  original clustering produced, was an artifact of its seed count and clustering radius, not a
  property of the arm. A general 6R has up to 16.
* Basins are roughly uniform: at one pose the 8 solutions were hit `[24, 24, 24, 22, 15, 15]`
  times out of 135 converged seeds.
* Rare solutions exist: one pose showed a 5th solution hit **once** in 800 seeds, so a census
  that stops when it stops finding new solutions can miss a small-basin solution. This is the
  gap that homotopy continuation (Finding 2) is meant to close.

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

### 3.5 The pair census: what the empirical criterion actually detects

`python3 -m study.exp02_pair_census --poses 5 --seeds 400 --resolve` (SR5; 118 pairs of
distinct solutions over five poses; every pair scanned with the refined-minimum crossing test)

* **Paths that touch the singular set: 112/118.** 68 pairs cross once, 44 cross *twice* (the
  straight line leaves the chamber and comes back), 0 graze. Only 6 pairs' straight paths avoid
  `Sigma` altogether, with clearances `5.5e-4 … 4.9e-3` against a `sigma_min` scale of `0.078`
  at the solutions — i.e. the paths that "avoid" the singular set pass within 0.7–6% of the
  typical clearance. Three of those six dip to a manipulability ratio of 0.003–0.005, which is
  exactly the "sharp dip, never zero" counterexample from the original experiments: a near miss,
  now measured rather than suspected.
* **The criterion as a crossing detector** (truth = the refined minimum of `sigma_min` reaches
  zero, which is what "the path crossed a singularity" means):

  | dip threshold | dips fired | true positives | false positives | false negatives |
  |---|---|---|---|---|
  | 0.20 / 0.10 / 0.05 | 118 | 112 | 6 | 0 |
  | 0.01 | 112 | 112 | 0 | 0 |
  | 0.005 | 100 | 100 | 0 | 12 |

  A looser threshold never misses a crossing but fires on near misses; a tighter one misses
  crossings, because a dip's depth depends on *which* singular stratum is crossed — the
  manipulability is a product of six singular values, and crossing a boundary where only one of
  them vanishes need not take the product near zero. On an earlier six-pose run the same table
  had 11 false negatives at a 0.01 threshold, so the safe threshold is arm- and pose-dependent:
  the criterion has no setting that is simultaneously exact everywhere, which is the honest
  form of "it holds with high probability in higher dimensions".
* The **one-Jacobian sign certificate** (`sign det J(q1) != sign det J(q2)`) fired for exactly
  the 68 odd-crossing pairs, with **zero disagreements** against the path scans. It is the cheap
  sufficient test the question asked for: two Jacobians and no path at all.

### 3.6 Distinct solutions can share a branch: the criterion's assumption is false

Same run, with clearance walks (`study/chamber.py`) at `delta = 5e-3`:

* **14 pairs were connected by a witnessed path keeping `sigma_min >= 5e-3`** — i.e. provably in
  the same connected component of `Q \ Sigma`. Six of them are the pairs whose straight path
  avoids `Sigma`; the other eight are pairs whose straight path crosses it twice, so the double
  crossing was a detour, not a branch boundary.
* **Negative control: 68/68 certified-different pairs could not be connected** by the same walk.
  The machinery is therefore self-consistent — and getting it there mattered: with endpoint-only
  step checks, 2 of 16 control pairs were "connected", which is how a false certificate looks.
  Every accepted step is now verified along its whole segment.
* Why this is a *proof* and not evidence: a path in `Q \ Sigma` from `q1` to `q2` with
  `FK(q1) = FK(q2) = T` projects to a loop in the workspace based at `T`, and the path *is* the
  lift of that loop. Two solutions joined that way are two points of one sheet of the covering
  `FK : Q_reg -> W`, i.e. one branch in the standard sense (the sheet, not the solution).
* So the branch structure of a 6R is **not** one branch per solution: the number of branches is
  the number of connected components of `Q \ Sigma`, and a component can cover the workspace
  with degree > 1 (several solutions over the same pose). The practical consequence for the
  criterion is that it is **sound but incomplete**: a path witnessed to avoid `Sigma` does prove
  "same branch", while *crossing* `Sigma` does not prove "different branch" — the path may leave
  a chamber and return, which is what 8 of the 14 witnesses did.
* Open: the full partition of a pose's solutions into branches needs a better connectivity
  oracle than a greedy walk (randomised walks or a planner in `{sigma_min >= delta}`), and the
  interesting follow-up is whether those components correspond to the classical
  shoulder/elbow/wrist sign labels. That is round 3's experiment.

### 3.7 A practical two-stage branch test, and how coarse it is

The certificates compose into a cheap first stage and an expensive second one:

1. **`sign det J` splits the solutions into two groups** and *proves* that a pair taken from
   different groups is in different branches. Measured on three poses (9, 11 and 12 solutions):
   the groups came out `4+/5-`, `6+/5-`, `6+/6-`, and of the `20`, `30`, `36` cross-group pairs,
   **none** was connectable by a clearance walk — consistent with the certificate.
2. **Within a group** the walk is what decides, and it is incomplete: of the `16`, `25`, `30`
   same-group pairs, `4`, `7`, `9` were certified connected (same branch), the rest unresolved.
   So the branch count is strictly smaller than the solution count — 12 solutions do not mean 12
   branches — but this walk alone cannot finish the partition. That is exp03's job.

### 3.8 A roadmap oracle, its audit, and why it is not yet the answer

`study/roadmap.py` samples configurations that keep clearance, joins the ones that can see each
other along a *length-verified* segment, and reads components off the graph.  Two things came out
of building it, both worth keeping:

* **A certificate defect, caught by the audit.** The first version verified every edge with a
  fixed sample *count*.  On a long edge that leaves gaps, and the roadmap promptly reported
  "17 solutions, one branch" for a pose whose own determinant signs prove at least two chambers.
  Edges are now sampled per *radian* of length (`SAMPLES_PER_RADIAN = 400`) and capped at
  `MAX_EDGE = 0.6` rad, and `audit_against_sign` re-checks every component against the one-Jacobian
  certificate as a standing test.  Measured after the fix: **0 violations** on three poses
  (8, 17 and 8 solutions).  This is the second time in this study that an endpoint-only clearance
  check produced a false certificate, which is why the audit is part of the module now.
* **The sampling is not yet dense enough to partition.** 300 points in a six-dimensional box of
  half-width ~4 rad give a road of isolated nodes: the partition came out as 6, 15 and 8
  components with 2 solutions unattached, i.e. mostly inconclusive.  The oracle is *sound* but
  *weak*, and the next iteration should grow trees from the solutions outward (bidirectional or
  RRT-style) instead of sampling the box uniformly.

One measurement from that run is worth recording on its own: the census at one pose produced 17
"distinct" solutions whose closest pair is **1.1e-04 rad** apart, while other poses have closest
pairs of 2.5 rad.  A 6R cannot have 17 solutions, so the census is producing numerical twins in an
ill-conditioned direction; the deduplication tolerance needs to follow the conditioning (or the
solutions need to be continued, not re-solved).  That is a census-quality issue, not a branch
issue, and it is now visible.

## 4. Finding 2: DH continuation

### 4.1 SR0 is exactly one parameter away, and the search says which one

`python3 -m study.exp10_sr0_geometry` measures the pairwise geometry of the joint axes
(`study/arm_geometry.py`) rather than reading solvability off three different DH notations.

* **SR5**: axes 1-2, 1-4, 4-5, 5-6 intersect; axes 2 and 3 are parallel (0.403 m apart); and the
  wrist's third pair, **4-6, is skew by 0.136 m** -- the last three axes do *not* meet, so Pieper
  does not apply. That distance is the measured form of "there is no closed form".
* **Exhaustive single-parameter relaxation** over all 6 links x 3 translation components: exactly
  one closes the wrist -- **link 5's y offset, +0.136 m -> 0**, after which 4-5, 5-6 *and* 4-6 all
  intersect. That is `SR0`, and the homotopy in §4.2 travels exactly 0.136 m.
* The same fingerprint identifies the ER3 (7-DOF) as a textbook S-R-S arm: {1,2,3} pairwise
  intersecting (spherical shoulder), {5,6,7} pairwise intersecting (spherical wrist) -- a check
  that the instrument reports the structure it should.

### 4.2 The first-order formula is O(step^2), and the corrector is what makes it usable

`python3 -m study.exp11_dh_continuation --steps 6`, SR5, well-conditioned solution
(`sigma_min = 0.114`), moving along the SR0 direction (`link 5 y`):

| step (m) | first order | +1 Newton | +3 Newton | exact IK (same solution) | error ratio per decade |
|---|---|---|---|---|---|
| 0.1 | 1.72e-02 | 2.93e-03 | 1.08e-11 | did not converge | - |
| 0.01 | 1.71e-04 | 1.81e-07 | 1.55e-14 | 3.08e-08 | 100.5 |
| 0.001 | 1.71e-06 | 7.52e-12 | 1.42e-16 | 1.24e-06 | 100.1 |
| 1e-04 | 1.71e-08 | 4.70e-15 | 4.70e-15 | 1.25e-08 | 100.0 |
| 1e-05 | 1.71e-10 | 1.28e-16 | 1.28e-16 | 1.69e-10 | 100.0 |

* The step-to-step error ratio is **100.0-100.5 per decade of step size**: the first-order
  prediction is second order, which is the quantitative form of the claim.
* **One Newton step takes the residual to ~0.2 * step^3** (1.71e-04 -> 1.81e-07 at 0.01 m); three
  steps reach machine precision.  A continuation should therefore always correct, and the
  predictor alone is only good for small steps: at 0.1 m it is 1.7 cm off.
* The continuation lands on the **same solution** a full IK solve finds on the perturbed arm:
  `|q_continued - q_exact|` is 4e-08 to 1e-10 for steps up to 0.01 m.  At 0.1 m the library IK
  fails to converge from the original seed, which is the argument for stepping incrementally
  rather than jumping.
* **Near the singular set the formula is defined but unusable**, as suspected: at
  `sigma_min = 2.96e-03` the same 0.01 m step produces `|dq| = 2.20 rad` and a 0.70 m residual,
  and three Newton steps still leave 2.0e-02 m.  The continuation needs a step controller that
  watches `sigma_min`, and the neighbourhood of a singularity is where solutions are created,
  destroyed or exchanged -- i.e. where the interesting part of the homotopy happens.

### 4.3 The SR0 solver: deterministic, complete, and validated

`study/sr0.py` solves SR0 without a random seed, using only structure measured from the model:

* the shoulder centre is the intersection of axes 1 and 2, constant at ``(0, 0, 0.328)``;
* the target pose gives the wrist centre, which sits at a fixed point of the flange,
  ``(0, 0, -0.1035)`` -- measured, not assumed;
* **q1**: the whole chain rotates about the base z axis through the shoulder, so rotating the
  wrist centre *back* by a candidate ``q1`` gives that branch's planar target; the two candidates
  are the azimuth and the azimuth plus pi.  Using ``(+radius, height)`` for both was the first
  bug: in the second branch the planar target's radial coordinate is *negative*, and every
  candidate then missed by up to 0.9 m;
* **q3**: measured property -- the wrist centre's distance from the shoulder depends on ``q3``
  alone -- so a scalar root-find over one turn is complete, and ``q2`` follows in closed form from
  the direction ``psi(q3)`` read off the model;
* the second bug was the *sign* of ``q2``'s in-plane rotation: axis 2's sense is a URDF property,
  and assuming the wrong one rejected every candidate silently.  It is now measured
  (``turn_sign = -1`` for this arm) with one forward-kinematics call;
* **the wrist**: measured ``n4 . n6 = cos(q5)`` exactly, so ``q5 = +-acos(n4 . n6)``, and a
  deterministic two-variable Newton finishes ``(q4, q6)``; every candidate is verified against the
  forward kinematics before it is returned.

Validation (`python3 -m study.exp12_sr0_solver --poses 5 --seeds 400 --match 1e-3`):

* **8 solutions at every pose**, worst pose residual ``8.0e-10``, ``72 ms`` per solve;
* against the multi-start numerical census: **0 solutions missed in either direction**;
* the census's count was *larger* than 8 at one pose (15), and all 15 matched the analytic 8
  within ``1e-3`` rad -- the numerical twins of §3.8 again.  A spherical-wrist 6R cannot have 15
  solutions, so this is a property of the census, not of the arm; the analytic solver is the
  reference here, and the census needs a conditioning-aware dedupe before it can be trusted to
  count.

### 4.4 The homotopy SR0 -> SR5, and what it does not reach

`python3 -m study.exp13_homotopy_sr0_to_sr5 --poses 2 --seeds 800 --retries 6`.  Seeds are SR0's
fibre from the deterministic solver (eight solutions); the tracker carries each along
``link 5 y: 0 -> 0.136 m`` with the predictor-corrector and a step controller on the clearance;
the reference is a 800-seed multi-start census of the real arm.  The parameter space is 18- or
19-dimensional, all eight seeds reached on the *straight* path (0.7 s for the whole pose).

* **Pose A: 8 of 8 arrived, and the sets agree exactly** -- 0 census solutions missed, 0 homotopy
  solutions the census missed.  Where the arm has eight solutions, the method is complete:
  SR0's closed-form fibre, carried 0.136 m, *is* the SR5's fibre.
* **Pose B: 6 of 8 arrived, census 13, and 8 census solutions were never reached** (`sigma_floor =
  2e-3` halts the other two paths at ``s = 0.72`` and ``s = 0.96``).  The straight path meets the
  discriminant there -- where a pair of solutions is born or dies -- and a tracker that cannot
  switch branches cannot follow a solution through it.  This is the measured form of "the formula
  is defined near the singular set but not usable" from 4.2, and the reason the missing solutions
  are exactly the ones behind such a meeting point.
* **Retries on generic curved paths** (the standard trick: add a bump that vanishes at both ends,
  so the endpoints stay SR0 and SR5 while the route changes) help but do not close the gap: with 6
  retries per stopped branch, pose B went from 5 to 6 arrivals.  The discriminant has codimension
  one, so a generic path avoids *most* of it, but a solution that does not exist at ``s = 1`` on
  any real path connected to the seed cannot be reached this way at all.
* One solution at pose B was found by the homotopy and *missed* by the 800-seed census, which is
  the same census under-counting seen in 3.1/3.8 from the other side.

So the honest statement of the method is: **continuation from a solvable neighbour gives a
guaranteed subset of the fibre -- complete when the two arms' solution counts match along the path
-- and completeness in general needs branch switching at the discriminant** (deflation, or
tracking the pair through the fold), not just a better step controller.

### 4.5 Next

1. ~~A closed-form solver for SR0~~ -- done above (deterministic and complete; the ``(q4, q6)``
   finish is a two-variable Newton rather than a closed form, and every solution is verified
   against the FK before it is returned).
2. An incremental homotopy SR0 -> SR5 following all eight SR0 solutions, with a step controller
   on `sigma_min` and branch switching where the path meets the discriminant; then count how many
   of SR5's solutions (up to 16) it reaches, against the multi-start census as reference.
3. Use the same tracker to walk loops in the *workspace* and get a complete census (the gap in
   section 3.1), which closes Finding 1 too.

## 5. Next

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
