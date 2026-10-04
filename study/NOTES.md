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

### 3.9 The classical branch labels do not label this arm's branches

`python3 -m study.exp15_branch_labels --poses 2 --seeds 500`, with `study/labels.py` computing the
textbook shoulder/elbow/wrist signs from the *axes* (each vanishes on one stratum of the singular
set: shoulder = wrist centre in the plane of axes 1-2, elbow = shoulder-elbow-wrist collinear,
wrist = axes 4 and 6 aligned), and `study/exp15_branch_labels.resolve` deciding pairs by directed
walks (both directions plus random admissible waypoints -- the one-directional walk of 3.7 leaves
most pairs unresolved).

Measured, and it refutes the naming:

* **Each of the four label triples that occur appears exactly twice, and the two solutions sharing a
  triple have opposite `det J` signs** -- so the sign certificate *proves* they are in different
  branches.  The labels are therefore at most half as fine as the branches.
* **12 pairs with different labels are connected by a witnessed path** (same branch), and **8 pairs
  with identical labels are not** (and are provably different by the sign certificate).  So the
  labels are neither necessary nor sufficient for "same branch" on this arm.
* The reason is structural rather than numerical: the classical names assume the singular strata
  they are named after.  The SR5's wrist is *offset* and its axes 2 and 3 are *parallel*, so its
  singular set does not consist of those three strata, and a triple product that does not vanish on
  any stratum cannot label branches.  The same caution applies to the SR4/SR5 class in general.
* The directed oracle works and stays consistent: **0 walks across sign groups** (the certificate
  audit), and the walk resolves more pairs than 3.7's single-direction version (7 of 28 at pose 0
  versus 4 of 28 there).

So the answer to "is there a cheap branch test" is:

1. **`sign det J`** -- one Jacobian per solution, *rigorous* for "different branch" (it fired on
   exactly the odd-crossing pairs, 0 disagreements over every experiment in this study), and
   silent otherwise;
2. **a witnessed clearance path** (walk, both directions, through waypoints) -- constructive for
   "same branch", incomplete in the negative direction;
3. **not the classical labels** -- measured false in both directions on this arm, which is worth
   knowing before anyone builds a branch classifier on them;
4. for the arms where an analytic fibre exists (SR0 here), the closed-form branches are the
   labels, and the homotopy of section 4 carries them to the arm that has no closed form.

### 3.10 (Finding 2) The lost solutions come from multiple roots splitting -- measured

`python3 -m study.exp16_root_splitting --census-seeds 350 --grid 21`.  One pose, the straight path
``link 5 y: 0 -> 0.136 m``, sweeping the fibre size of the *intermediate* arm and looking at what the
stalls actually are:

* **Every stall is a double root.**  The stalled configuration coincides with another real solution
  of the same intermediate arm, at distances ``1.5e-6``, ``4.0e-5`` and ``9.4e-5`` rad.  The tracked
  branch does not "run out of iterations" -- it runs into a *coalescing pair*.
* **The fibre size is not monotone, and it changes at the stall points**: 8 solutions up to
  ``s = 0.65``, then **16** at ``s = 0.70``, 14 at 0.75, 12 at 0.80, 15 at 0.95 and **10** at
  ``s = 1``.  Across each stall the count *drops* (17 -> 14, 18 -> 12, 18 -> 10 over ``+/-0.02``):
  those are **death events where a pair annihilates**.  Between ``0.65`` and ``0.70`` the count
  *doubles*: **birth events**, which no tracked branch visits because they happen to other pairs.
* The lost solutions are exactly that: of the real arm's 12, the five tracks cover five, and the
  **seven lost ones all have healthy clearance (0.009-0.065) and sit 0.7-4.9 rad away from every
  reached solution** -- they are different branches, not numerical ghosts, and they were born at
  events the path never visited.
* So the mechanism is the one the "multiple root" intuition predicts: at the discriminant the fibre's
  real count jumps, a tracker that follows a single branch sees only its own death (when it is the
  one dying) and never the births; §4.5's reseeding fired only at *stalls*, i.e. only at deaths,
  which is why a residue survived.

The completion this points at: **scan the path for events by fibre size, not by stalls, and reseed at
every event** (births included).  That is the next experiment.

### 3.11 (Finding 2) Scanning the path instead of following branches: complete coverage

`python3 -m study.exp17_event_reseed --pose-seeds 0 1 --grid 17 --census-seeds 250`.  The consequence
of 3.10: stop following branches and scan the path.  At every sample of ``s`` the intermediate arm is
solved from scratch; every solution found anywhere along the sweep becomes a seed and is carried
forward to the real arm; the reference is a 1000-seed census of the real arm.

* **Pose 0** (the one whose fibre jumps ``8 -> 16 -> 14 -> 12 -> 13 -> 13 -> 10``): 164 distinct seeds
  over 17 samples, 10 distinct arrivals, reference census 11 -- **0 reference solutions missed, 0
  arrivals the census missed**.  (The 11th census "solution" is the twin artifact of 3.8/4.3: it
  matches one of the ten within ``1e-3``.)
* **Pose 1**: fibre ``6 -> 5`` along the path, 93 seeds, 6 arrivals, reference 5 -- **0 missed**, and
  **1 arrival the census did not find**: the analytic-seeded continuation is again more complete than
  random-restart IK.
* Price, measured: the scan is 50-71 s per pose (17 samples x 250 seeds) plus 7-15 s of tracking,
  against ~1 s for following the eight SR0 branches and ~10-20 s for a 1000-seed census.

So the method is complete in the practical sense that matters -- it covers a thorough census and
exceeds it -- at scan cost, and the reason it had to become a scan is exactly 3.10: births happen to
pairs no tracked branch is following.

### 3.12 The redundant case: franka's fibre, self-motion, and how often distinct means different

`python3 -m study.exp18_franka_redundant --seeds 200` plus the pair walk below.  On a 6-DOF arm the
two things people mean by "branch" coincide, so the distinction never comes up: solutions of one pose
are on one sheet exactly when a path between them avoids ``Sigma``.  On a 7-DOF arm they come apart,
and the measurements are:

* **The fibre is a continuum.**  One pose (200 seeds): **36 solutions**, clearances ``0.089-0.205``
  (`study/census.py` clusters them, but "count" means "clusters found", not "isolated solutions").
* **Self-motion paths between solutions are real and roomy.**  Integrating the Jacobian's null-space
  direction *with a pose correction after every step* keeps ``FK(q) = T`` to ``1e-15`` while
  travelling tens of radians of joint space, and the clearance along those traces stays above
  ``6.5e-2``.  So solutions along such a trace are in **one chamber, connected with a healthy
  margin, without the pose ever changing** -- the empirical criterion's premise survives (its
  covering argument is dimension-free), but the practical question changes completely: at a redundant
  pose most "different solutions" are not different branches at all.
* **The methodology matters**: without the pose correction the trace *drifts*, and a first version of
  this experiment reported 27 "components" of length 30 rad that never closed.  Measured drift is now
  reported per component (``worst pose drift`` ~ ``1e-15``) -- the third time in this study that an
  unverified step produced a confident, wrong number.
* **Component counting is not yet structural.**  Traces ran the full step budget (600 steps x 0.05 rad
  per direction) without closing except one (605 points, closed), so the returned 15 "components" are
  a budget artifact, not a manifold measurement.  Recorded as open; the fix is loop closure or
  arc-length clustering rather than more steps.
* **Distinct solutions are much more often same-chamber here.**  Witnessed clearance walk
  (`study/chamber.py`, ``delta = 5e-3``) over 20 random pairs: **franka 7/20 connected**, SR5 control
  in the same run **1/20**.  Small samples, clear direction -- and it is the same phenomenon as the
  fourteen same-chamber pairs of 1.4, only much more frequent.

### 3.13 One fibre's partition, determined: two branches, and the cheap certificate is enough

`python3 -m study.exp19_partition --poses 3 --seeds 600 --waypoints 6` (plus one earlier run at 800
seeds).  Two certificates applied to the same fibre -- ``sign det J`` (rigorous for "different", one
Jacobian per solution) and a witnessed clearance route (constructive for "same", tried both ways and
through random admissible waypoints) -- and then the equivalence classes by transitive closure:

| pose | solutions | sign groups | pairs: same / different / undetermined | classes after closure | dip-heuristic false positives |
|---|---|---|---|---|---|
| 1 | 8 | 4+ / 4- | 7 / 16 / 5 | **[4, 4]** -- the sign split | - |
| 2 | 8 | 4+ / 4- | 7 / 16 / 5 | [4, 3, 1] | 4 |
| 3 | **12** | 6+ / 6- | 26 / 36 / 4 | **[6, 6]** -- the sign split | 4 |
| 4 | 8 | 3+ / 5- | 10 / 15 / 3 | **[5, 3]** -- the sign split | 2 |

* **The partition is the determinant-sign split in three of the four poses**, i.e. the O(1)
  certificate is not just sound but *sufficient* there -- and the pair-level residue does not matter,
  because the witnessed pairs already generate the classes by transitivity.
* The 12-solution pose has **two** branches of six, not twelve and not eight: "number of solutions"
  and "number of branches" differ by a factor of six on this arm.
* The remaining pose leaves one sign group of four as ``[3, 1]``: either it holds two branches or the
  walk missed a route.  That is the honest residue of this method, and it is the only one in four
  poses.
* The **dip heuristic's false positives are now certified**: 2-4 pairs per pose dip below ``0.01``
  although they sit inside a transitively *proved* same-branch class.  Compare the measured error
  rates of 1.3: the near-miss false positives are not an artefact of a threshold choice, they are
  routes that come close to the singular set without reaching it.

### 3.14 The classical labels do not survive redundancy either -- measured along self-motion

`python3 -m study.exp20_labels_redundant --robot franka --seeds 150`.  On a redundant arm the test is
sharper than 3.9's: follow a **self-motion** path (pose constant) that **keeps clearance**, so every
configuration on it is in one chamber *by construction*, and watch the three textbook signs.  Any flip
along such a path proves that sign is not an invariant of the chamber.

* 26 solutions at the pose, only **4 distinct label triples**; the elbow sign was ``+1`` for all of
  them.
* **4 of 6 tested solutions change their label triple along the path**: e.g.
  ``(1, 1, -1) -> (1, 1, 1)`` after **12.00 rad** of travel with ``min clearance 9.48e-02`` and pose
  drift ``6.0e-15``; another ``(-1, 1, 1) -> (-1, 1, -1)`` at clearance ``7.09e-02``.
* So on this arm the labels are not even chamber invariants -- they change *inside* a single chamber.
  3.9 had them merely half as fine as the branches; redundancy makes it strictly worse, and any
  classifier built on the shoulder/elbow/wrist triple is unsound here in the strongest sense.
* What the 12 rad of travel with a healthy margin also shows: a single chamber of this arm is *large*.
  The right coordinates for the redundant case are the chamber plus the position along the
  self-motion manifold -- the labels are not those coordinates.

### 3.15 The redundant fibre's components, with the limits handled: a lower bound of 13

`python3 -m study.exp18_franka_redundant --seeds 150`, now with the two corrections 3.12 was missing:
the trace stops at **joint limits** (`within_limits`) and every component reports *why* it ended, so a
budget artefact can no longer masquerade as a manifold fact.

* 26 solutions, **13 components** at a 0.02 rad membership tolerance, sizes
  ``[5, 6, 1, 1, 1, 1, 1, 3, 2, 1, 2, 1, 1]``.
* The end reasons are now informative: some components are **short arcs terminating at a joint limit**
  (0.54 and 0.78 rad), one **closes on itself** after 2.57 rad, and the rest are long arcs (30-60 rad)
  that either end at a limit or run out of the step budget ("end" empty).
* So **13 is a lower bound, not a count**: three components are unresolved by budget and could merge
  with others if traced further.  What is solid: the clearance along every traced path stays above
  ``7.1e-02`` and the pose drift below ``6.8e-13``, so all of these really are self-motion paths with
  room to spare, and the fibre is *large* -- tens of radians of travel per component.
* Together with 3.14 (labels change inside a chamber) and 3.12 (35% of random solution pairs are
  witnessed same-chamber), the redundant picture is: one pose has many solutions joined by long,
  generous self-motion arcs and by cross-component routes, so "which solution" and "which branch" are
  much weaker notions than on a 6-DOF arm.

### 3.16 The endgame: the multiple root is computed, and it splits like a square root

`python3 -m study.exp21_endgame --pose-seed 0`.  Instead of walking around the discriminant, solve
for it: at a fold the pose is right *and* the Jacobian is singular, which is a square system in
``(q, s)`` -- six pose equations plus ``det J = 0``, seven unknowns -- solved by Newton from the
stall.

* **The folds are found exactly**: the three stalls refine to ``s* = 0.718354``, ``0.961848``,
  ``0.799244`` with augmented residuals ``3.2e-11``, ``9.0e-13``, ``6.5e-11``.  The multiple root is
  now a computed object, not an observation.
* **How the branches split away from it**, measured on a geometric sequence of gaps
  (``s* - 10^-k``, each re-solved from the previous configuration -- an adaptive tracker's own step
  halving hides the exponent):

  | fold | distances to ``q*`` at gaps 1e-2 .. 1e-6 | fitted exponent |
  |---|---|---|
  | 0.718354 | 0.197, 0.0669, 0.0223, 0.0097, 0.0072 | 0.44 |
  | 0.961848 | 0.115, 0.0377, 0.0137, 0.0082, 0.0074 | 0.39 |
  | 0.799244 | 0.191, 0.0678, 0.0229, 0.0107, 0.0085 | 0.42 |

  The decay per decade is a factor **2.8-3.1** over the unsaturated points, i.e. a power law with
  exponent **~0.45-0.48**: ``q(s) = q* + c (s* - s)^(1/2)``.  That is the **square-root splitting of
  a double root -- two branches per fold** -- and it is far from ``1/4`` or ``1/3``, which is what a
  fourfold or threefold root would give.
* Two honest caveats, both measured: the last sampled decade saturates (the re-solve cannot resolve
  the branch closer than ~7e-3 rad, so those points are dropped by the fit), and the fitted exponent
  sits systematically ~8% below 1/2 -- consistent with ``q*`` being the nearest singular point rather
  than this branch's exact coalescence, whose linear offset flattens the slope.
* **Corollary to test next**: since every fold splits *two* branches, the observed jump of the fibre
  size from 8 to 16 cannot be one high-order root producing eight branches -- it must be **four
  separate folds**.  Counting the distinct folds along the path would settle it.

### 3.17 The fold-count corollary: not settled, because this instrument is not good enough

The corollary from 3.16 was that a jump of the fibre size from 8 to 16 must be *four* folds, since
each fold splits two branches.  `python3 -m study.exp22_fold_census --pose-seed 0 --seeds 150` tried to
count them by locating the jump with bisection on the fibre size and then counting the coalescing
(near-singular) configurations just after it.  It does not work, and the reason is worth recording:

* **The instrument is a statistical estimate.**  At a fixed ``s`` the 150-seed census returns 18 and
  then 16 on neighbouring samples; the count fluctuates by +-2, so "where the count changes" is not a
  well-defined event and the bisection refines noise.
* **Near-singular clusters at a grid point are not folds.**  The clusters found (3, 10, 8 at the three
  apparent events, with clearances from 6e-4 to 2e-2) count what the census happened to find at that
  ``s``, which includes configurations that are merely close to the singular set -- the same near-miss
  population that produced the dip heuristic's false positives in 1.3.
* So the corollary **remains unverified**.  The instrument that would settle it is not a bigger census
  but a different one: follow the zeros of ``det J`` **as functions of ``s``** (a fold curve per
  coalescing pair, tracked like any other branch, with exp21's augmented system as the corrector) and
  count how many distinct curves the path crosses.  That is a well-posed 1-D problem, unlike counting
  a noisy fibre size.

Recorded as an open item rather than a result, in line with the three earlier instrument failures
(3.3/3.8/4.2): the numbers a method produces are only as good as what the method actually measures.

### 3.18 Structural completeness: the fold points are isolated, and the 8 -> 16 jump is four of them

exp23's first design was wrong in an instructive way: it tried to *track* fold curves in ``s``, but for
a fixed target pose the augmented system (7 equations, 7 unknowns) has **isolated** solutions, not
curves -- the tracker therefore stood still, refining the same point 61 times while the clearance at
its ends stayed at ``1e-11``, which is how the mistake was visible.  The right count is the number of
distinct fold *points* in an interval, and the right seeds are deterministic: the near-singular
configurations **collected along the tracked branches**, not a random census (exp22's failure).

Measured (pose seed 0, ``link 5 y: 0 -> 0.136 m``, 121 near-singular configurations collected, each
refined by the augmented Newton, deduplicated at ``1e-4`` in ``s``):

| fold ``s*`` | clearance at the fold |
|---|---|
| **0.655086** | 8.8e-12 |
| **0.656878** | 3.6e-11 |
| **0.657375** | 2.1e-11 |
| **0.658119** | 1.5e-10 |
| 0.718354 | 1.1e-10 |
| 0.799244 | 1.7e-12 |
| 0.961848 | 2.2e-10 |

* **The 8 -> 16 jump is exactly four folds**, bunched inside a window of ``0.003`` in the parameter
  (0.6551 .. 0.6581), each a double root: ``4 x 2 = 8`` new solutions, which is the observed jump.
  So the "multiple root splitting" picture is right in mechanism and *twofold* in multiplicity -- the
  jump comes from four independent double roots, not one root of higher order.
* The three later folds are the deaths found in exp21 (``0.718354``, ``0.799244``, ``0.961848``),
  reproduced here to six decimals from an independent seed set: the two instruments agree.
* **Structural completeness** for this pose: every solution of the real arm can now be attributed to a
  birth at one of the four early folds and, where applicable, a death at a later one -- not merely
  "the scan finds everything" (exp17) but *where each solution comes from*.
* Methodological note, the fourth of its kind in this study: the deterministic seed set (tracked
  branches) worked where the statistical one (a 150-seed census, exp22) did not.  Structure beats
  sampling for locating events, exactly as the branch certificates beat the dip heuristic.

### 3.19 The "two solutions per fold" law: direction confirmed, magnitude unresolved by this count

`python3 -m study.exp24_fold_law --poses 3 --seeds 400` tests
``fibre(s + eps) - fibre(s - eps) = +-2 x (folds in the window)`` over several poses, with folds found
deterministically (near-singular configurations on the tracked branches, refined by the augmented
Newton) and the fibre size measured by a 400-seed census.

| pose | folds | groups | measurement | predicted |
|---|---|---|---|---|
| 0 | 8 | 0.6551 (4 folds) | 8 -> 18, delta +10 (birth) | 8 |
| 0 | | 0.7184 (1) | 17 -> 14, delta -3 (death) | 2 |
| 0 | | 0.7992 (1) | 17 -> 12, delta -5 (death) | 2 |
| 0 | | 0.9618 (2) | 16 -> 10, delta -6 (death) | 4 |
| 1, 2 | 0 | - | no near-singular configuration on any tracked branch | - |

* **The direction is right**: the fourfold group *adds* solutions (birth) and every single/double
  group *removes* them (death), which is the mechanism exp21 measured; the two poses with no folds
  also have no events.
* **The magnitude is not resolvable with this instrument.**  The census-based count carries the +-2
  spread measured in 3.17/3.8 (under-counting and twins), which is exactly the size of the signal
  (2 per fold): deltas of 10, 3, 5 and 6 against predictions 8, 2, 2 and 4 are all the prediction plus
  up to one unit of count noise per endpoint.
* So the law needs a **deterministic** fibre count -- continuity-based, tracking every solution from
  one ``s`` to the next (the scan of exp17, or a tracker seeded densely) -- rather than a statistical
  one.  Recorded as the next instrument, in line with 3.17: the same mistake twice would be a habit.

### 3.20 The deterministic count: the law holds exactly, and the exception refines it

`python3 -m study.exp25_deterministic_count --pose-seed 0` splits the fibre into **survivors** (every
previous solution carried forward by the predictor-corrector -- exact, no sampling) and **births**
(solutions of the new arm matching no survivor), so the count difference is deterministic up to birth
detection:

| window | folds | survivors | died | born | delta | prediction |
|---|---|---|---|---|---|---|
| [0.650, 0.662] | 4 | 8/8 | 0 | **8** | **+8** | +8 |
| [0.710, 0.725] | 1 | 14/16 | **2** | 0 | **-2** | 2 |
| [0.792, 0.806] | 1 | 12/14 | **2** | 0 | **-2** | 2 |
| [0.950, 0.975] | 2 | 10/12 | **2** | 0 | **-2** | 4 |

* **Three windows obey the law exactly and with no noise at all**: the fourfold group adds exactly
  eight solutions (eight births, no deaths), and each single fold removes exactly two.  The census
  noise that defeated exp24 is gone because nothing is being estimated.
* The fourth window is the interesting one.  It contains two folds but loses only two solutions, so
  one of the folds does not change the real fibre size -- and that is not a defect of the count but a
  property of folds: a fold only changes the *real* count when its coalescing pair crosses between
  real and complex.  Probed at ``s* +- 5e-3`` for the fold at ``0.961848``, the nearest real solution
  is ``8.2e-2`` rad before and ``3.0`` rad after -- the dying pair is real before and gone after,
  while the other fold of the window has no real pair near it on either side: a complex turning point.
* **Refined law**: ``fibre`` changes by ``+-2`` per *real event*, i.e. per fold whose pair changes
  reality; folds of complex pairs deform the fibre without changing what is real.  Counting folds is
  therefore an upper bound on the events, which is exactly why exp24's "2 folds, expected 4" was
  wrong in its premise rather than in its arithmetic.

### 3.21 Trying to predict the events independently: half the law is exact, the other half is not

exp26 set out to test the refined law over several poses by *predicting* each fold's kind (birth,
death, or invisible turning point) and comparing with the deterministic count.  The prediction failed,
and the failure localises exactly where the instrument is weak:

| window | folds (predicted kinds) | deaths | births | measured delta |
|---|---|---|---|---|
| [0.650, 0.663] | 4 (all "invisible") | **0** | **9** | +9 |
| [0.713, 0.723] | 1 ("invisible") | **2** | 0 | -2 |
| [0.794, 0.804] | 1 ("invisible") | **2** | 0 | -2 |
| [0.957, 0.967] | 2 ("death","death") | **2** | 0 | -2 |

* **The deaths are exact and they are the law's solid half**: every window's death count comes from
  *tracking* (a track either arrives or it does not), and the two death windows give exactly 2 deaths
  each for one fold, the fourfold window gives 0 -- no noise anywhere.
* **The classification is the broken part.**  Probing the fibre at ``s* +- 5e-3`` for a solution within
  a fixed 0.1 rad of the fold asks the census to find a pair sitting at ``c sqrt(eps)`` from the fold,
  with ``c`` unknown and the basins near a fold small: it reported "invisible" for folds that
  demonstrably change the count.  The right classifier tracks the pair itself (seeded at the fold with
  the square-root radius from 3.16) instead of sampling for it -- the fifth time in this study that
  sampling lost to structure, and the same lesson as 3.17/3.19/3.20.
* **The births are census-based and therefore noisy**: the fourfold window measures 9 births where the
  structure predicts 8 (4 folds x 2).  Deaths are exact because tracking decides them; births are
  "solutions the census found that match no survivor", so they inherit the census's +-1.
* Net: the law is **verified exactly on the death side** (4 windows, 0 discrepancies) and **to within
  one solution on the birth side**, with the remaining error traced to the birth detector rather than
  to the law.  Predicting *which* folds are events independently is still open, and the instrument for
  it is now clear.

### 3.22 Tracking the pair: the law verified exactly in three windows, and the fourth localises a blind spot

exp27 classifies a fold by *tracking its pair* instead of sampling: the pair separates along the
Jacobian's null direction at the fold, and it sits at ``c sqrt(eps)``, so a local solve seeded at
``q* + c sqrt(eps) v`` answers a well-posed question.  With that classifier the law of 3.20 is tested
again (exp26 now delegates to it):

| window | folds | classified | predicted | measured | verdict |
|---|---|---|---|---|---|
| [0.650, 0.663] | 4 | birth x4 | **+8** | 8 -> 16 (0 died, 8 born) | **agrees** |
| [0.713, 0.723] | 1 | death | **-2** | 16 -> 14 (2 died) | **agrees** |
| [0.794, 0.804] | 1 | death | **-2** | 14 -> 12 (2 died) | **agrees** |
| [0.957, 0.967] | 2 | death x2 | -4 | 12 -> 10 (2 died) | disagrees |

* **The classifier works, and it confirms the square-root law per pair**: the distances it finds at
  gaps ``1e-4, 1e-3, 5e-3`` are ``0.0119, 0.0374, 0.0822`` -- ratios ``3.14`` and ``2.20`` against
  ``sqrt(10) = 3.16`` and ``sqrt(5) = 2.24`` -- for a fold pair, and ``0.0374, 0.119, 0.266``
  (``3.18``, ``2.24``) for a birth pair.  Independent of 3.16's fit, and per pair.
* **Three windows agree exactly**, including the fourfold birth window: four births predicted, eight
  solutions gained, zero deaths -- the mechanism of 3.10/3.16/3.20 in one line.
* The fourth window's disagreement is now localised precisely.  The two folds there are **not**
  duplicates: their ``s*`` agrees to ``1e-10`` but their configurations are **8.89 rad apart**, so they
  are two genuinely different fold points that happen to be born at the same parameter value.  Both
  are classified "death", yet only one removes solutions -- and the classifier cannot tell the two
  apart, because it asks *whether* a real solution is near the fold rather than *how many*: a death
  has one solution (the pair) on the near side and none beyond, while a **turning point has two on
  both sides**.  Counting the pair instead of testing for its presence is the fix, and it is the only
  thing between this law and a clean four-for-four.

### 3.23 Counting the pair: three windows exact, and the fourth narrows to a radius question

exp27's classifier now counts how many real solutions sit near the fold on each side (seeded along
``+-v`` with amplitudes ``c sqrt(eps)``), because *how many* is what separates a death (one on the
near side, none beyond) from a turning point (two on both sides):

| window | classified | predicted | measured | verdict |
|---|---|---|---|---|
| [0.650, 0.663] | birth x4 | +8 | 8 -> 16 (0 died, 8 born) | **agrees** |
| [0.713, 0.723] | death | -2 | 16 -> 14 (2 died) | **agrees** |
| [0.794, 0.804] | death | -2 | 14 -> 12 (2 died) | **agrees** |
| [0.957, 0.967] | death x2 | -4 | 12 -> 10 (2 died) | disagrees |

* The birth window is now exact, mechanism and count: four folds, four births, eight solutions gained,
  zero deaths -- and each birth pair independently obeys ``c sqrt(eps)`` with ratios 3.18 and 2.24
  against ``sqrt(10)`` and ``sqrt(5)``.
* The fourth window still disagrees, and the reason is now specific.  Its two folds are 8.89 rad apart
  in configuration space (same ``s*`` to 1e-10), and the second one reads "death" although the count
  loses only two solutions.  With a search radius of ``8 sqrt(eps) + 1e-3 ~ 0.57 rad`` and twelve
  solutions spread over the fibre, the solutions the classifier finds "near" that fold need not be the
  pair at all -- they can be unrelated neighbours that happen to be inside the radius, which would make
  the near side read as occupied for a fold whose own pair is complex on both sides.
* The fix is the same principle as everywhere else in this study: size the neighbourhood by the
  *measured* law instead of a generous constant.  3.22 measured ``c ~ 1.2`` for these pairs, so the
  radius should be ``~1.5 c sqrt(eps) ~ 0.09 rad`` at ``eps = 5e-3``, not 0.57 -- and a solution found
  inside a tight radius should then be confirmed by tracking it a short step in ``s``.
* Status: the law stands exactly where the instrument is tight (three windows, including the birth
  window that started this whole line of work); the one disagreement is now a stated measurement
  problem with a stated fix, not an unexplained residual.

### 3.24 The classifier is radius-sensitive in both directions, and the reason is fundamental

Tightening the pair-tracking neighbourhood to the measured law (``1.5 c sqrt(eps)`` up, as 3.23
proposed) broke the windows that were previously exact:

| window | tight radius | generous radius (8 c sqrt(eps)) | measured |
|---|---|---|---|
| [0.650, 0.663], 4 folds | all "complex fold" (predicted -8) | all "birth" (predicted +8) | **+8** |
| [0.713, 0.723], 1 fold | "complex fold" (-2) | "death" (-2) | -2 |
| [0.794, 0.804], 1 fold | "complex fold" (-2) | "death" (-2) | -2 |
| [0.957, 0.967], 2 folds | "death x2" (-4) | "death x2" (-4) | -2 |

* The tight radius cannot find the pair at all (the birth folds read as complex), the generous one
  finds unrelated neighbours (the fourth window's turning point reads as a death).  Neither is right,
  and the reason is not a badly chosen constant: **solving for a configuration within
  ``c sqrt(eps)`` of a fold is ill-conditioned by construction** -- the pair is precisely the
  degenerate direction -- so both "which solutions are there" and "how many" are hard to establish by
  local solves.
* What *is* reliable is tracking: a branch carried across the window either arrives or it does not, and
  that is how the deaths were measured exactly in 3.21-3.23 (four windows, zero discrepancies).  So
  the deterministic count is not merely a check on the classifier -- it **is** the classifier, and the
  independent per-fold prediction is the part that remains open.
* Reverted to the generous radius (which gives three of four windows) with the asymmetry documented in
  the docstring, rather than leaving a tighter constant that looks principled and performs worse.
* Sixth methodological instance of the same shape: the instrument, not the phenomenon, decides what
  the numbers mean -- and here the instrument's limit is a conditioning property, not a tuning
  parameter.

### 3.25 Four ways to solve the same pose: what to use, and what not to

`python3 -m study.exp28_solver_comparison --poses 3` compares the library's own solver, the multi-start
census, the SR0 homotopy on the straight path, and the SR0 scan -- no privileged reference, each judged
by what it misses of the union and what it costs.

| pose | library (20 seeds) | census (300 seeds) | homotopy (8 SR0 seeds) | scan (9x200) | union |
|---|---|---|---|---|---|
| 0 | **0** (0.1 s) | 8 (1.9 s) | 8 (0.7 s) | 8 (25 s) | 12 |
| 1 | **0** (0.1 s) | 8 (2.0 s) | **0** (0.4 s) | 8 (32 s) | 8 |
| 2 | **1** (0.1 s) | 8 (1.9 s) | 8 (0.5 s) | 8 (29 s) | 8 |

* **The library's `IkStandard` finds nothing usable from random seeds on this arm** -- one solution in
  three poses at twenty seeds each, which matches the baseline of 0.0-0.2% random-seed convergence
  measured earlier.  For the SR5, "call the IK solver" is not a working strategy; it needs a seed
  within about 0.05 rad (which is what the earlier basin measurements showed).
* **The multi-start census is the best value**: 24 solutions over the three poses in ~2 s each, i.e.
  complete on two poses and 8 of 12 on the third.
* **The SR0 homotopy ties it at a quarter of the cost** (0.5-0.7 s, 16 solutions) when the straight
  path carries its branches; it returns nothing when every branch stalls (pose 1), which is the
  discriminant case of 3.10 -- so it is a fast path, not a guarantee.
* **The scan buys completeness at 15x the census's cost** and, in this sample, found nothing the
  census missed -- its advantage appeared in 3.11/3.17 on poses where the census under-counted, i.e.
  it is insurance for the hard poses rather than the default.
* Practical ordering for this arm: census (~2 s) as the workhorse; SR0 homotopy (~0.5 s) when speed
  matters and the path is unobstructed; the scan when completeness is the point; and never the plain
  library solver from a random seed.

### 3.26 The recipe as one call, and what the three modes actually deliver

`study/complete_ik.py` packages the study's method behind a single entry point, ``solve_all(model,
target, mode=...)``, which returns deduplicated solutions each verified against the forward kinematics
(``max(position error, rotation error) <= 1e-6``, which is the library's own acceptance scale).
Its self-check on two poses (different poses from exp28's stream, which is why the numbers differ):

| pose | ``census`` (300 seeds) | ``homotopy`` (8 closed-form seeds) | ``scan`` (9 grid points) |
|---|---|---|---|
| 0 | **3** (3.2 s) | 4 (1.1 s) | **8** (43 s) |
| 1 | **6** (2.1 s) | **0** (0.0 s) | **8** (56 s) |

* **Only the scan reached eight on both poses.**  The census returned 3 and 6 -- the same under-counting
  measured in 3.8/3.14/3.17 and in the solver comparison of exp28, where the *same* mode found 8 on
  other poses: how much a 300-seed census finds is pose-dependent, and it can be less than half.
* **The homotopy's zero on pose 1 is not a bug but its characteristic**: branches that stall at a
  discriminant do not arrive, so the fast mode returns nothing rather than something wrong.
* Worst residuals: ``2.1e-08`` for the homotopy and scan modes, ``4.0e-07`` and ``9.2e-07`` for the
  census (the census's solvers are accepted at the library's tolerance).  A caller gets the residual
  with every solution, so the difference is visible instead of implied.
* Practical reading: the scan is the mode to reach for when the *count* matters (it is the one that was
  complete in every experiment of this study), the homotopy when a sub-second answer on an
  unobstructed path is enough, and the census when a fast, partial set is acceptable.

### 3.27 How cheap can the complete mode be?  A budget sweep of the scan

`python3 -m study.exp29_scan_budget --pose-seeds 0` sweeps the scan's two knobs (grid points along the
path, seeds per grid point) on a hard pose:

| grid x seeds | solutions | time | misses of the richest budget |
|---|---|---|---|
| 3 x 60 | 6 | 5.1 s | 2 of 8 |
| **5 x 80** | **8** | **10.1 s** | **0 of 8** |
| 5 x 200 | 8 | 17.4 s | 0 of 8 |
| 9 x 120 | 8 | 22.3 s | 0 of 8 |
| 9 x 300 (the module's default) | 8 | 43.1 s | 0 of 8 |
| 15 x 200 | 8 | 51.9 s | 0 of 8 |

* **Five grid points with 80 seeds each reach everything the richest budget reaches, in 10 s instead
  of 43** -- a factor of four for free on this pose, and 15 x 200 buys nothing at all.
* The failure mode of too small a budget is graceful and visible: 3 x 60 finds 6 of 8 and takes 5 s, so
  a caller can start cheap and escalate when the count disagrees with a second method.
* The caveat matters and is measured elsewhere in this study: a budget that saturates on one pose is
  not a guarantee across poses (exp28 had a pose whose *union* over methods reached 12 while any single
  method found 8), so the calibration says "start at 5 x 80, and treat a disagreement with the census
  as the signal to escalate", not "5 x 80 is complete".
* Adding this as the documented default in :mod:`study.complete_ik` would cut the complete mode's cost
  fourfold; the module keeps the larger default for now, because the sweep is one pose and the point of
  the complete mode is completeness rather than speed.

### 3.28 Spending the scan's budget where the events are

`python3 -m study.exp30_adaptive_scan --pose-seeds 0 7` replaces the uniform grid by a two-pass one: a
coarse grid with a small census, then **two extra samples in every interval whose solution counts
differ** (the trigger is sound because a birth or death inside an interval shows up as different counts
at its ends), then the same carry-forward and deduplication.

| pose | coarse counts | adaptive | uniform | agreement |
|---|---|---|---|---|
| seed 0 | ``[4, 4, 4, 4, 8]`` | **8 solutions, 13.5 s** | 8 solutions, 42.8 s | 0 missed either way |
| seed 7 | ``[8, 8, 8, 8, 7]`` | **8 solutions, 9.7 s** | 8 solutions, 32.9 s | 0 missed either way |

* **Identical solution sets at a third of the cost** (3.2x and 3.4x), which is the point: exp29's
  calibration said most of the uniform budget is wasted, and targeting the intervals that change
  recovers it without losing anything on these poses.
* The coarse counts are themselves informative: pose 0's fibre grows from 4 to 8 across the path
  (births, as in 3.10), and the trigger fired exactly on the last interval of the grid.
* Caveat, unchanged from 3.27: two poses do not make a guarantee; the trigger can only refine an
  interval whose *ends* differ, so an interval with one birth and one death inside (net zero) would be
  missed.  Tightening that needs the fold spectrum of 3.18 rather than counts.

### 3.28.1 The adaptive scan's comparison basis, stated to avoid a misleading table

The reproduction archive of the final commit makes the nuance visible: with the *cheapest saturating*
uniform budget (``grid 5 x 80``, which exp29 identified) the uniform scan takes **10.3 s** against the
adaptive scan's **13.7 s** on the same pose -- the adaptive version pays for its extra refined samples.
Its advantage is against the *default* uniform budget (``grid 9 x 300``, 43 s), which exp29 showed buys
nothing over 5 x 80.  So the honest statement is:

* the adaptive scan is **not** faster than a well-chosen uniform budget; it is faster than the
  *default* one, and it decides where to spend without a calibration run;
* picking the uniform budget by hand (5 x 80 for this arm, 10 s) is the cheapest option measured
  anywhere in this study, at the price of not adapting to a pose with more events.

### 3.29 The fold trigger is sound but not economical -- measured, and the count trigger wins

exp31 replaces exp30's count trigger by the fold spectrum (3.18), which closes the "one birth and one
death in the same interval" hole because every event *is* a fold.  Three scans, same poses:

| pose | fold spectrum | fold-triggered | count-triggered | uniform |
|---|---|---|---|---|
| seed 0 | 4 folds at 0.9911-0.9936, found in **39.5 s** | 8 solutions, 18.9 s (9 samples) | 8 solutions, 13.5 s | 8 solutions, 43.1 s |
| seed 7 | **none**, found in 4.2 s | 8 solutions, **6.3 s** (5 samples) | 8 solutions, 9.7 s | 8 solutions, 33.0 s |

* **All three find the same eight solutions on both poses** (0 missed in every direction), so the
  trigger changes the cost, not the answer -- which is the property a refinement trigger should have.
* **The fold spectrum costs more than it saves**: 39.5 s to enumerate the folds of pose 0 (about forty
  near-singular candidates, each refined by the augmented Newton at ~106 ms), against which the 18.9 s
  scan is a detail; the total (58 s) is worse than the uniform scan it was meant to beat.
* **Where it does win is the event-free pose**: with no folds to trigger on, the fold-triggered scan is
  just the coarse grid and comes in at 6.3 s, the cheapest scan measured anywhere in this study -- so
  the method's cost adapts to the pose's event structure, which the uniform grid cannot do.
* Practical reading: use the **count trigger** (13.5 s) as the default refinement -- cheap and
  sufficient on these poses -- and compute the spectrum when it is wanted for its own sake (it is what
  gives the birth attribution of 3.18).  Paying 40 s for a trigger that saves 5 s is not a trade; the
  "one birth and one death in one interval" hole is better closed by taking the coarse grid to six or
  seven points, which costs a census and not a spectrum.

### 3.30 Instrument failure 8: averaging a configuration-dependent quantity (and franka is not S-R-S)

The user asked whether the franka's last three axes really intersect, pointing at a figure in the
franka repository.  They do not, and the way the wrong claim got into this study is worth recording.

* **What was claimed and why it was wrong.**  In a discussion turn I stated that franka's axes 5, 6, 7
  pairwise intersect (a spherical wrist).  That number came from the **ER3** fingerprint measured
  earlier in this study -- ER3 *is* a textbook S-R-S arm -- and I applied it to franka without
  measuring franka.  My own franka repository documented the opposite all along
  (``docs/api.md``: row 7 ``[0.088, 0.0, 1.5708, 0.0]`` "(the wrist offset)";
  ``docs/method.md``: "wrist offset | DH row 7, a = 0.088"), and the repository's method removes that
  offset as its first step.  So the correction is not new information to the project, only to this
  study's summary.
* **The tool defect that would have caught it.**  ``arm_geometry.axis_fingerprint`` averaged the
  pairwise axis distances over random configurations.  That is harmless for **adjacent** axes --
  axis ``i`` is fixed in link ``i-1``, axis ``i+1`` in link ``i``, and the two links differ by a
  rotation *about axis i*, which leaves it in place, so the pair distance is structural -- but for
  ``j >= i+2`` the intervening joints move one axis relative to the other and the distance is a
  property of the configuration.  Averaging it reports a number no configuration has:
  the same table printed ``5-7: 0.000`` while the axis lines it had just printed were 88 mm apart,
  which is the self-contradiction that exposed the defect.
* **Fix.**  ``axis_fingerprint`` now measures at one configuration (zeros by default) and says so;
  ``structural_report`` prints only invariants: adjacent-axis distances and, for a triple, the
  distance from the intersection point of the first two axes to the third (concurrency).  The
  user-provided URDF makes the franka fact unambiguous: ``<joint name="panda_joint7">`` has
  ``<origin xyz="0.088 0 0">`` and ``<axis xyz="0 0 1"/>``, so the joint-7 axis is displaced 88 mm
  from the 5-6 intersection point.
* **Corrected table (all entries configuration independent):**

  | arm | shoulder 1-2-3 | wrist, last three axes |
  |---|---|---|
  | franka panda | **concurrent, 0.0000 mm** | **not concurrent, offset 88.0000 mm** |
  | ROKAE ER3 (7 axes) | concurrent | **concurrent (true S-R-S)** |
  | SR5 | not concurrent | not concurrent, offset **136.0000 mm** |
  | SR0 (link 5 y = 0) | not concurrent | **concurrent, 0.0000 mm** |

* **What changes and what does not.**  franka is **not** S-R-S: its *shoulder* is spherical and its
  *wrist* carries an 88 mm offset, which is exactly why my earlier franka work had to reduce the
  offset away before applying an elbow law.  Nothing in Finding 2 changes, and it is now *stronger*:
  SR0's sphericity is confirmed by the invariant test (0.0000 mm) rather than by an average over
  random poses, and SR5's obstruction is exactly 136.0000 mm.
* **Lesson, the eighth of its kind.**  Two errors in one: quoting a measurement from a different
  robot, and a tool that averaged a quantity that is not constant.  Both were caught only by a
  reader comparing a claim against a figure; the defence that should have worked is in the study's
  own conventions -- *every claim with the command that produces it* -- since running the franka
  fingerprint would have shown 88 mm immediately.

### 3.31 Uniqueness domains: the definition, corrected, and what it changes

The study's `READING.md` (and the handover prompt, and both AIs' citations of it) wrote Wenger's
characteristic surfaces as ``CS_i = f^-1(f(A_i*)) cap A_i`` without saying what ``A_i*`` is.
The review says it plainly (arXiv:1610.04080 section 11, fetched as the arXiv PDF and read in full):

> "Let `A_i*` be **the boundary of aspect `A_i`**. The characteristic surfaces `{CS_i}` associated
> with `A_i` are `{CS_i} = f^-1(f(A_i*)) cap A_i` ... since an aspect is defined as an open set,
> `A_i` does not contain its boundary i.e. `A_i* cap A_i = empty` thus `{CS_i}` might be empty
> (note that if this is the case for all its aspects, **the robot is not cuspidal**)."

So ``CS_i`` is the preimage of the image of *this* aspect's boundary, not "where the other aspects
overlap". Three consequences, all of which the study uses from here on:

1. ``CS_i \subset f^-1(Delta)`` with ``Delta = f(Sigma)``: **crossing a characteristic surface is a
   pose landing on the discriminant image**, which is exactly the event a whole-fibre tracker can
   see (two branches collide).
2. "empty for every aspect" is an equivalent characterisation of *non*-cuspidality, next to the one
   the study has been using ("an aspect holding two solutions of one pose").
3. The components of ``A_i \ CS_i`` are uniqueness domains (`Ra_ij`), ``f`` is one-to-one on each
   (Wenger 2004), and their images are the *regions of feasible paths*; the maximal ones are
   ``Qu = A_i - C(Ra_ij)`` (aspect minus the closure of one ``Ra``). For a **non**-cuspidal robot
   the aspects themselves are the uniqueness domains and ``f(A_i)`` are the feasible regions -- the
   statement the study had been implicitly assuming for SR5.

Corollary the SR5 measurement needs: if aspect ``A_i`` holds ``m`` solutions over a pose, those
``m`` solutions lie in ``m`` *different* uniqueness domains (injectivity). So the 12-solution pose
of exp19 is 2 aspects **and 6 uniqueness domains per aspect**, and "same aspect" does not imply
"one trackable Cartesian region".

### 3.32 A cuspidal arm that is solvable by radicals: exp46 repaired, witness measured

`python3 -m study.exp48_cuspidal3r` (arm in `study/cuspidal3r.py`, position-task 3R orthogonal from
Wenger's Fig. 1, MDH rows ``[d, alpha, a]``).

exp46 failed for two independent reasons, both now repaired and both worth recording:

* **The task was wrong.**  It built a 6-DOF pose from ``fk`` of a random configuration and ran a
  pose-IK census on a 3-joint arm. A 3-joint arm's image is 3-dimensional in SE(3), so that pose
  has a **one-point** fibre: the "census" measured nothing (0 solutions over the whole grid, which
  exp46 read as a rank problem). The task must be the *position*.
* **The family was degenerate**, exactly as exp46's own docstring diagnosed: ``Rx(-90)`` composed
  with ``Rx(90)`` leaves joint 3's axis parallel to joint 2, so ``rank J = 2`` everywhere.
* A third, silent one: ``RobotModelNumpy.build(param_type, solver, config)`` does **not** read
  ``base``/``ee`` from the config mapping -- they are function arguments. Both exp46 and the first
  version of the probe put them in the config and silently got the identity tool transform.

Measured (all numbers from the run; the arm is `[d,alpha,a]` rows ``(0,0,0), (1,pi/2,1),
(2,pi/2,0)`` with tool row ``(1.5,pi/2,0)``):

| phase | measurement |
|---|---|
| rank | 200 random configurations: rank-3 fraction **1.00**, min rank **3**, worst ``sigma_min`` 5.16e-04 |
| aspects | ``T^2 \ Sigma`` labelled by flood fill at 300x300: **exactly 2** components, cell counts 44700/44700, singular cells 0.0067 |
| fibres | 40 poses: 25 with 2 solutions (per-aspect multiplicities (1,1)), 15 with 4 solutions (**2,2**) |
| cuspidality | an aspect holds **two solutions of one pose** => cuspidal, by definition, with no planner and no budget |
| workspace | 25x25 tool points: 4 solutions 110 (18%), 2 solutions 310 (50%), unreachable 205 (33%) |
| radicals | the fibre's ``tan(q3/2)`` are the roots of a monic quartic whose coefficients agree to **4.2e-13** across three azimuths => the IK reduces to a quartic => solvable by radicals (Ferrari; the review cites Kholi & Spanos 1985 for the coefficients) |

Two analytic facts used by the instrument, both verified numerically (residual 2.7e-09):
``det J = -rho * det(Dg)``, where ``Dg`` is the Jacobian of the reduced map ``(q2,q3) -> (rho,z)``
and ``rho`` is the tool's distance from the base axis; and the determinant does not depend on
``q1``, which is why the aspect labelling is a two-dimensional problem.

**Open (recorded, not resolved):** the arm's cusp was *not* localised. A census-based triple-root
cluster sits near ``rho ~ 2.6, z ~ +-2.2`` (three solutions mutually within 0.14 rad), but the
"spread of the three closest solutions" statistic is dominated by census noise and a pattern
search did not drive it down. The instrument that should settle it is the discriminant of the
quartic (a zero set in ``(rho^2, z)``) or the study's own augmented system -- not a census.

### 3.33 Instrument failure 9: a fibre tracker without a step bound, and a witness pose rebuilt from a printed number

Two failures from the same session, both caught by their own controls.

* **Seeding Newton with the previous configuration is not continuation.**  The first version of
  `study/taskspace.py` predicted nothing: each sample's corrector started from the previous
  configuration. On a closed loop this produced ``min gap = 6e-15`` between two tracks (they merged
  into one solution) and reported "21 of 125 loops permute the fibre, none identity" -- a whole
  phase of numbers that were the tracker, not the arm. Fixes, now defaults: the predictor is the
  **tangent** step ``dq = pinv(J) Delta target``; a correction further than ``jump_tolerance``
  (0.3 rad) from the prediction is rejected; a **step longer than ``max_step``** (0.5 rad) is
  rejected because near the discriminant ``pinv(J)`` is huge and an unbounded step is exactly how a
  tracker leaves its branch while still reporting a tiny residual; two live tracks closer than
  ``merge`` (1e-8) are flagged as a tracker fault, distinctly from a fold; and the degenerate loop
  (all targets equal) is a standing control that must be the identity with zero collisions.
  This is the same failure mode as Q17, reproduced from the other side -- and it is a reminder that
  a tracker's *positive* results need a negative control more than its negative ones do.
* **Do not reuse a printed number as an input.**  Phase 4/5 of the first exp48 rebuilt the witness
  pose from its printed ``(rho, z)`` (three decimals). The rebuilt point sat on the discriminant
  image -- a census found **0 solutions in 100 starts** there and Newton stalled at 3e-4 -- so the
  entire measurement was about a pose with no solutions at all. The fix is procedural: keep the
  full task vector, print only for reading.

The same failure mode, one instrument later (`exp49`): on SR5 the whole-fibre lift of a witness
loop tracked only 1 of 10 branches to the end, and the *diagnostic that distinguishes a fold death
from a tracker death* -- ``sigma_min`` at the last configuration plus the distance to the nearest
other live track -- says tracker: the stops sit at ``sigma_min`` 5e-3 to 1.7e-2 (healthy) with the
nearest other branch 0.22 to 14 rad away, and **zero collision samples** in the whole loop
(shortest gap 2.2e-1). A fold death is a pair that first collides and then vanishes; none of these
did. The positive half of the same run is solid and is corroborated independently: the tracked
branch arrives at a *different* solution of the same pose, with joint-space distance **0.0e+00**,
along a joint path whose clearance stays at 5.25e-3 -- and a route from one solution to another is
a valid lift of the loop, so arriving exactly at the second solution is not something a drifting
tracker produces by chance. So "a prescribed Cartesian loop can change posture on SR5 without
meeting a singularity" is measured; "the loop's image crosses the discriminant image" is *not*
measured this round, and the fix direction is stated (the corrector's error must be
``se3_error(fk(q), T_target)`` rather than a difference of two errors against a fixed reference,
which is only a first-order approximation, plus a task-step controller on ``sigma_min``).

The same ambiguity appears in `exp48` phase 4, where it is stated in the output rather than
papered over: the certified route joins two solutions of one aspect, so the lift of its projection
*must* permute them, and the tracker reports the identity. Since phase 2's cuspidality claim is a
label count on the torus (no tracker involved), it stands; the lift does not.


**The fix, and what it bought (same round).**  The residual must be measured *relative to the
current pose*: ``se3_error(FK(q), T)`` is what ``homotopy.jacobian_q`` differentiates, whereas
differencing two errors against a fixed reference pose is only a first-order approximation of it
and leaves the correction direction wrong by a pose-dependent linear map.  With that change, a
rigid-target interpolation for bisection (``taskspace.se3_interpolate``: slerp in rotation, linear
in position, endpoints exact to 1e-16, orthonormality to 1e-16) and the max-step guard, the same
``exp49`` run now *shows the crossing*:

* **3 collision samples of 118** (samples 46, 47, 48), with two tracks merging to **9.12e-15 rad**
  and dying together -- tracks 0 and 7, whose last accepted configuration has ``sigma_min`` 8.5e-3
  with the nearest other branch 17.5 rad away.  A pair that collides and then vanishes *is* a fold,
  i.e. the loop's image landing on the discriminant: the "two sheets die at a fold" signature,
  measured rather than argued;
* the tracked pair still arrives (track 1 -> solution 6, joint-space distance **0.0e+00**) and the
  degenerate-loop control stays **10/10 arrivals, 0 collisions, 0 merged**.

So on SR5: a prescribed Cartesian loop changes posture without meeting a singularity *and* its
image crosses the discriminant image -- the posture change is a change of **uniqueness domain**, not
of aspect.  What remains instrument-limited: six branches still stop at healthy clearance with no
partner (tracks 2,4,5,8 at sample 41, 6 at 48, 3 at 0) -- tracker deaths, not folds, and the debt
is stated rather than averaged into the result.


**Reproduction (three pose seeds, `exp49 --pose-seed 0|1|2`).**  The posture change reproduces on
all three; the *crossing* was caught on one.

| pose seed | fibre | same-aspect pair (clearance) | tracked posture arrives at | arrivals | crossing collisions |
|---|---|---|---|---|---|
| 0 | 10 (5+/5-) | (1, 6), 5.25e-3 | solution 6, **0.0e+00** | 1/10 | **3** (a pair merging to 9.12e-15 and dying together) |
| 1 | 8 (4+/4-) | (0, 2), 5.07e-3 | solution 2, **0.0e+00** | 5/8 | 0 (shortest gap 7.1e-2) |
| 2 | 8 (4+/4-) | (0, 2), 6.40e-3 | solution 2, **0.0e+00** | 2/8 | 0 (shortest gap 9.2e-2) |

Seed 1 and 2 also show the print defect that was fixed with them: an arrival with no matching
starting solution was printed as "arrived at solution 3 (distance 1.4e+01 rad)", which reads like a
permutation and is not one.

**Honest limit of the crossing evidence.**  No collision on seeds 1/2 does *not* mean no crossing:
the detector only sees a crossing when the coalescing pair is **real at the base pose and tracked**,
and the samples must straddle it. A crossing between two samples, or one whose pair was complex at
the base pose (a birth), is invisible to a forward tracker -- the same blindness that exp25 had to
patch with reseeding. So the reproduced claim is the *posture change as a prescribed Cartesian
loop* (3/3); the *crossing* is measured on 1/3 and the sampling limitation is the stated reason,
not a negative result.


### 3.34 The A→B answer in numbers: one prescribed loop, ten postures, ten different feasible arcs

The fate table of exp49's seed-0 run *is* the answer to the planning question, read per posture: the
same prescribed Cartesian loop (118 samples, the projection of a certified same-aspect path) is
trackable in full by exactly one posture of the pair that the loop joins, and by nobody else.

| posture | samples of the loop it can track | share |
|---|---|---|
| 1 (the tracked one, arrives at solution 6) | 118/118 | **100%** |
| 0, 7 (a pair that collides at 9.12e-15 and dies together) | 48/118 | 41% |
| 2, 6 | 45/118 | 38% |
| 4, 5, 8 | 41/118 | 35% |
| 3 | 0/118 | 0% |

(track 0: 48/118 = 41%, track 1: 118/118 = 100%, track 2: 45/118 = 38%, track 3: 0/118 = 0%, track 4: 41/118 = 35%, track 5: 41/118 = 35%, track 6: 45/118 = 38%, track 7: 48/118 = 41%, track 8: 41/118 = 35%; command: `python3 -m study.exp49_sr5_uniqueness --pose-seed 0 --seeds 400`.)

Read the table the way the theory says to read it: the loop's image is a *single* curve in the
workspace, and each posture owns a different initial segment of it. So "the robot can follow this
path" has no posture-free answer, and the useful statement for offline programming is exactly the
per-posture one -- the *feasible arc*, whose end is where that posture's branch meets the
discriminant image. Combined with the reproduction of the posture change (3/3, NOTES 3.33) this is
the operational form of "the regions of feasible paths are the images of the uniqueness domains":
the loop leaves the starting posture's feasible region at the sample where its branch dies, and it
can only be completed by a posture change -- which is what the tracked pair does, non-singularly.

Planning rule that follows (and that the instrument can test directly): for a prescribed Cartesian
path and a chosen posture, track the whole fibre along the path; the posture is usable while its own
track survives, and the events that end that are the collisions of two *other* branches with each
other (a pair dying at a fold) -- the path's image crossing the discriminant image, i.e. leaving the
uniqueness domain it started in.


### 3.35 Q21 probed and refuted: the max-step guard is not what discards the six branches

Q21's working hypothesis was that the six branches exp49 drops *at healthy clearance with no partner*
(know Q21) were casualties of the corrector's step bound.  Measured, on the same seed-0 run with the
bound relaxed tenfold (``--max-step 5.0`` against the default 0.5):

| run | arrivals | deaths | collision samples | shortest gap | tracked pair |
|---|---|---|---|---|---|
| ``--max-step 0.5`` | 2/10 | same six | 3 (46, 47, 48) | 9.12e-15 | track 1 -> solution 6, 0.0e+00 |
| ``--max-step 5.0`` | 2/10 | same six | 3 (46, 47, 48) | 6.69e-14 | track 1 -> solution 6, 0.0e+00 |

Identical in every reported quantity, and the identity control stays 10/10 in both.  So the step
bound is *not* the cause: whatever stops those six branches happens inside the corrector's
convergence at a task step the bisection could not shrink usefully, not at the guard.  The
hypothesis is recorded as refuted rather than quietly dropped -- a fix aimed at the guard would
have changed nothing, and the probe cost one run.

What the probe does *not* distinguish (and what the next instrument should): whether the six
branches genuinely cease to be real inside a sample interval (a fold the samples straddle) or the
corrector lands in a neighbouring basin whose residual happens to be near zero.  The instrument for
that is Q20's birth reseeding plus a per-sample double-root test, not a bigger step bound.


**The same table on two more poses (from the seed-1/seed-2 logs of the three-seed reproduction).**
The per-posture feasible arc is not an artefact of the seed-0 pose: on seed 1 (8 solutions, 102
loop samples) the tracked posture walks 102/102 = **100%** while the other branches stop at
51-52/102 = 50-51%; on seed 2 (125 samples) the tracked posture is 125/125 = **100%** while the
others stop between 10/125 = 8% and 70/125 = 56%.  Every stop is again at healthy clearance
(``sigma_min`` 5.7e-3 to 2.2e-2) with the nearest other branch 7e-2 to 4.2e+01 rad away, i.e. the
same instrument-limited class as in 3.35, not folds.  So the shape of the answer is stable across
poses: **one posture owns the whole prescribed loop, every other posture owns a proper initial
segment of it.**


**Grid-resolution check on the exact aspect count.**  ``exp48``'s cuspidality claim rests on
labelling ``T^2 \ Sigma``, so the count must not be a grid artefact:

| grid | aspects | cells per component | singular cells |
|---|---|---|---|
| 200 | **2** | 19798 / 19798 | 1.01% |
| 300 | **2** | 44700 / 44700 | 0.67% |
| 500 | **2** | 124500 / 124500 | 0.40% |

Two components at three resolutions, with the two cell counts equal to each other to the cell (the
arm is symmetric under ``q3 -> -q3``) and the singular share falling as the grid refines -- the
signature of a *curve* of singular cells rather than a thickening.  Command:
``python3 -c "from study import cuspidal3r as c3; m=c3.build(); print(c3.aspect_labels(m, grid=500)[0].max())"``.


**How big is the job Q18 has to finish?  The aspect multiplicity distribution.**  The uniqueness
domains of an aspect are at least as many as the aspect's multiplicity over a pose (``f`` is
injective on each, so ``m`` solutions over one pose need ``m`` domains), which makes the
distribution the size of the "complete partition" task.  Measured with 300-seed censuses on twelve
poses (`model.sr5`, seeds 11):

| fibre | per-sign-group multiplicities | poses |
|---|---|---|
| 12 | (6, 6) | 1 |
| 8 | (4, 4) | 8 |
| 6 | (3, 3) | 1 |
| 4 | (2, 2) | 1 |
| 2 | (1, 1) | 1 |

So the *typical* SR5 pose has 8 solutions in two groups of four -- four uniqueness domains per
aspect -- and the 12-solution pose that exp19/exp49 use is the richest case, not the common one.
The sign split is exactly the aspect split in all twelve poses here (consistent with exp19's 12/13),
so Q18's remaining work is: resolve one aspect into its ``m`` uniqueness domains for the poses that
matter, with ``m`` between 1 and 6.


**A structural cusp candidate, and the fibre test that refuted it.**  The census-based triple-root
statistic (3.32) was noise-dominated, so the cusp was hunted structurally instead: a cusp of the
singular *image* is where the image velocity along the critical curve vanishes, i.e. where
``Dg . t = 0`` with ``t`` the tangent of ``{det Dg = 0}``.  Over 240x240 samples of the critical
curve the criterion separates cleanly -- median ``|image velocity|`` **1.596**, smallest
**7.157e-03** (ratio 4.5e-03) -- and a pattern search refines the winner to

    q2 = +2.25147559, q3 = +2.98854297  ->  (rho, z) = (3.46013884, +1.43635918),  |v| = 2.14e-03

with a mirror at ``z = -1.4364`` (two cusps, matching the review's 0/2/4 possibilities).

**The fibre test does not confirm it.**  At the candidate and at three inward probes
``(rho - 2e-3, 8e-3, 3e-2)`` the census finds **2** solutions and no triple (spread ``inf``), where a
cusp must show three coincident solutions.  So the vanishing image velocity there has another
cause -- the critical curve's image is *tangent* to itself or to the coordinate fold ``rho >= 0``
without a triple root -- and the criterion alone is not sufficient.  The fibre count is the arbiter;
the 3R cusp therefore remains **unresolved**, now with a refuted candidate rather than a noisy
guess.  Next instrument: the discriminant of the quartic (a zero set in ``(rho^2, z)``), which is
structural on both sides, plus a triple-root test seeded *at* the candidate with the pair-tracking
amplitude that exp27 measured (``c sqrt(eps)``).


### 3.36 The posture change is two-way: the reversed loop maps the pair back, to 1e-12

The A-to-B claim needs its other half: traversing the same prescribed loop backwards must take the
partner posture back to the first.  Measured on the seed-0 pose (the pair search happens to pick
(0, 7) here; the scratch probe reuses ``exp49``'s ``rich_pose`` and ``pose_kinematics`` verbatim):

| direction | arrivals | collision samples | shortest gap | pair |
|---|---|---|---|---|
| forward | 4/10 | 30 | 7.60e-15 | track 0 -> **solution 7**, distance **5.2e-12** |
| reversed | 2/10 | 43 | 4.97e-15 | track 7 -> **solution 0**, distance **6.6e-12** |

So the loop is a genuine *permutation of the fibre* and not a one-way drift: the two lifts are
mutual inverses to 1e-12, which is the signature a sheet exchange must have.  Two more things the
control shows and that are worth keeping: the number of collision samples depends on the pair (30
and 43 here against 3 for the (1, 6) pair of 3.33), i.e. *which* branch is being carried changes
where the images cross the discriminant; and the deaths are direction-asymmetric in the expected
way (forward, track 0 arrives and 7 stops at sample 31; reversed, 7 arrives and 0 stops at 44).

Command (scratch, not a study module -- it reuses ``exp49``'s functions):
``PYTHONPATH=<repo> python3 -u /home/yang/workspace/research/.scratch/reverse_probe.py``.


**A failed reproduction, and the reason is the planner, not the arm.**  Re-running the two-way probe
(3.36) on pose seed 1 aborts before any lift: the same-sign pair search finds **no pair that the
greedy clearance walk can connect**, so ``pair`` is ``None`` and the probe cannot start.  The cause
is the pose pool, not the physics: the scratch probe calls ``rich_pose`` with its default
``least=10``, while ``exp49`` reaches seed 1 only with ``--least 8 --tries 14`` (SR5 poses with ten
solutions are rare).  The walk's failure is the same *planner* incompleteness exp19 recorded
(waypoint budget, greedy stalls), which is why the 3R work replaced it with an exact label-grid
route -- on SR5 no such exact route exists yet, so **every SR5 pair statement in this study inherits
the walk's incompleteness**.  Two consequences worth keeping:

* the two-way result of 3.36 stands for the pose it was measured on, and its reproduction on other
  poses is **blocked by the pair search**, not shown to fail;
* the fix is not a longer walk but the same structural move used on the 3R: resolve the pair with a
  *certified* object (an exact route in a labelled region, or a collision-verified lift), and treat
  a walk failure as "未决" rather than as "not connected" -- which is what the study already does for
  the partition but not yet for the pair search inside the probes.


**The two-way result reproduces on a second pose, once the pose pool is the one exp49 uses.**  The
seed-1 attempt failed earlier for a stated reason (the probe's default ``least=10`` pool left the
same-sign pair search with nothing the greedy walk could connect).  With the pool widened to what
``exp49`` actually uses (``least=8, tries=14``) the probe runs and gives:

| pose | direction | arrivals | collision samples | pair |
|---|---|---|---|---|
| 0 | forward / reversed | 4/10 / 2/10 | 30 / 43 | 0 -> 7 at 5.2e-12 / 7 -> 0 at 6.6e-12 |
| 1 | forward / reversed | 5/8 / 3/8 | 0 / 0 | 0 -> 2 at 2.7e-12 / 2 -> 0 at 2.0e-12 |

So the permutation is mutual-inverse to ~1e-12 on **two** poses (and the forward half agrees with
the independent ``exp49`` run at seed 1, which reported track 0 -> solution 2).  Pose 1 also carries
the honest limit forward: **0 collision samples in both directions** -- the crossing was not caught
there, which is Q20's sampling blindness, not an absence of crossing.

Lesson, fourth of its kind and worth the line: the failed reproduction was a **pool** artefact, and
the diagnosis was only possible because the probe printed which stage stopped it.  A probe that
silently returns "no pair" would have read as a negative result about the arm.


### 3.37 Q21 answered, and an earlier reading of mine corrected: every stop is a fold

`python3 -m study.exp50_births_and_stops --pose-seed 0` (new module; the protocol registered as
Q23).  It tracks the whole fibre along the witness loop, then asks the *fibre* about every stop: at
the target where a track stopped, are there still real solutions within 0.2 rad of the stopped
configuration?  Seeds are perturbations of that configuration (200 of them, 0.3 rad box), so the
question is asked where the answer lives.

| track | stopped at sample | ``sigma_min`` there | nearest other branch | solutions near it | at the *next* target | verdict |
|---|---|---|---|---|---|---|
| 1 | 31 | 1.19e-02 | 2.47e-01 | 1 | **0** | fold |
| 2 | 45 | 1.39e-02 | 3.91e-01 | 1 | **0** | fold |
| 5 | 45 | 1.46e-02 | 3.91e-01 | 1 | **0** | fold |
| 7 | 31 | 1.16e-02 | 1.20e-13 | 1 | **0** | fold |
| 8 | 45 | 1.18e-02 | 4.46e-01 | 1 | **0** | fold |
| 9 | 31 | 1.16e-02 | 1.20e-13 | 0 | **0** | fold |

**6 folds, 0 tracker drops.**  That is a different answer from the one recorded in 3.33/3.35, and
the correction is the point: I had read "healthy ``sigma_min`` + no partner nearby" as *tracker
death*, and that reading was wrong.  The branch is perfectly real at the sample where it stops; what
is real at the *next* sample is nothing -- the pair coalesces and vanishes *between* the two samples,
which is exactly why the last accepted configuration sits at ``sigma_min ~ 1e-2`` instead of at zero.
The instrument that decides this is a local census at the next target, not the clearance at the stop.

Two consequences, both registered:

* **Q21 is closed**: the whole-fibre lift is not dropping branches; the branches die, and the six
  "no partner" stops are folds whose coalescence the sampling straddles.  The remaining instrument
  debt is not robustness at healthy clearance but *event resolution*, i.e. Q20.
* **Q20 stays open and is sharpened**: the gap detector flags **30 of 167 samples** with a pairwise
  gap below 1e-2 (branches approach each other over many samples around a fold), while the strict
  signature -- two tracks colliding to ~1e-15 and then dying together -- is rare.  What a forward
  tracker cannot see is a **birth** (a pair becoming real), which is why the seeds 1 and 2 runs
  reported zero collisions: their crossings may be births rather than deaths.  The fix remains
  per-sample reseeding (Q23 step 2), the one piece of exp50 not yet implemented.

Controls in the same run: the degenerate loop gives 10/10 arrivals, 0 permutations, 0 collisions,
0 stops -- the events belong to the arm.

Also recorded: a flood fill that precomputes its list of start cells labels 3480 components on a
two-component torus (the list is computed before any labelling, so every free cell starts a new
component). Cheap to fix, invisible without a sanity check on the component count -- which is why
the run prints the component count and the cell-count balance (44700/44700) rather than just the
label field.

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

### 4.5 Reseeding at the discriminant closes the gap -- and beats the census

`python3 -m study.exp14_reseed_at_discriminant --poses 2 --seeds 800`.  The observation that makes
the gap closable: the *target pose never moves*, only the DH parameters do, so at every ``s`` the
current arm has a whole fibre for that pose.  When a tracked branch dies at the discriminant, the
branches born at an earlier one are present at that same ``s`` -- a multi-start solve of the
*intermediate* arm finds them, and they can be carried on to ``s = 1``.

* **Pose B (the hard one, exp13: 5 of 8 arrived, 8 census solutions missed)**: reseeding at the two
  stalls started 53 further tracks, and the final count of census solutions still missed fell to
  **1 of 11**.
* **Pose A** (the earlier easy one) now shows the opposite error: the straight path arrived 8 of 8
  while an 800-seed census found only 5, so **6 of the homotopy's solutions were ones the census
  missed**.  At equal or lower cost the analytic-seeded homotopy is *more* complete than
  random-restart numerical IK -- which is the practical claim the method should be judged on, and
  it is the same census under-counting measured in 3.1, 3.8 and 4.4 from other directions.
* Tracker cost, for scale: the whole straight-path stage is ~1 s per pose; reseeding adds one
  local census per stall.

So the recipe that comes out of this is: **SR0's closed-form fibre -> track along the parameter
path -> reseed at every stall -> track again**, with the residual gap being the solutions whose
birth fold was never visited by the path.  Closing that last one needs an endgame (deflation) or a
complex path; the measurements above say how small that residue is in practice.

### 4.6 Next

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
