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


**Seed-1 reproduction of exp50** (``python3 -m study.exp50_births_and_stops --pose-seed 1``): 8
solutions (4+/4-), pair (0, 2), witness 103 configurations; the lift arrives 5/8 with live counts
8 -> 5, shortest gap **7.06e-02**, crossing candidates **0 of 102** samples, and the three stops are
classified **3 fold(s), 0 tracker drop(s)** -- tracks 4 and 7 have two real solutions near them at
the sample where they stop and none at the next target, track 6 has none at either.  The degenerate
control is 8/8 arrivals, 0 permutations, 0 collisions, 0 stops.  So 3.37's verdict -- every stop on
this loop is a branch that genuinely dies between two samples, not one the tracker lost -- reproduces
on a second pose, and Q21's closure is not a single-pose artefact.  It also shows how large the Q20
blind spot can be: this pose reports **zero** crossing candidates even though the tracked pair
changes sheet, so the death/birth events there are invisible to a gap threshold -- the reseeding
step is what has to find them.


**Seed-2 reproduction, and the three-pose tally.**  ``exp50 --pose-seed 2``: 8 solutions (4+/4-),
pair (0, 2), 126-sample witness; the lift arrives 2/8 with live counts 8 -> 2, shortest gap
**9.19e-02**, crossing candidates **0 of 125**, and the six stops are **6 fold(s), 0 tracker
drop(s)** (tracks 1, 2, 3 have one real solution near them at the stop and none at the next target;
tracks 4, 6, 7 have none at either).  Degenerate control: 8/8, no collisions, no stops.

Three-pose tally of the stop verdict: **15 stops, 15 folds, 0 tracker drops** (6 + 3 + 6).  So the
whole-fibre lift is not losing branches at any of the three measured poses, Q21's closure is
reproducible, and the corrected reading in 3.37 stands: a stop at healthy clearance means the pair
coalesces *between* samples, not that the corrector left the branch.  The same three runs also
measure Q20's blind spot consistently: seeds 1 and 2 report **zero** crossing candidates at all
(102 and 125 samples) even though their tracked pairs change sheet, so a gap threshold cannot find
those events -- per-sample reseeding is the instrument that must.


**Q23 step 2, minimal form, measured.**  ``exp50 --pose-seed 0`` now also reseeds **every sample**
from scratch (``--sample-seeds``) and reports the solutions that **no carried branch occupies** --
the births a forward lift cannot see, which is Q20's blind spot in its purest form:

    samples with unoccupied solutions: 45 of 168; total 253
    first samples 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134

So along this loop the carried branches do **not** cover the fibre at those samples: solutions exist
there for which the lift never had a track.  That is the mechanism Q20 names, now measured directly
instead of inferred from a missing collision -- the same loop whose sheet change exp49 measures has
solutions appearing that no track follows, and a detector built only on "two live tracks collide and
die" cannot register them.  Turning these into *tracked* births (adding them to the carried set and
following them forward, which is what exp25 did along the DH parameter) is the registered remainder.


**Bookkeeping hazard, recorded because it cost a duplicate commit.**  Two background recorder jobs
ended up patching the same file: one was launched in a round whose shell call was moved to the
background, so its "apply patch, then commit" ran *after* the same patch had already been applied
and committed by the next round's call.  Result: a redundant commit with an identical message, and
the same paragraph inserted twice into this file (both removed by the cleanup commit that carries
this note).  The guards written into the patchers ("insert only if the heading is absent") did not
catch it because the two versions of the patch used slightly different headings -- a guard keyed on
the *content* rather than on the operation is the fix.  Practical rule for this study: a recorder
must never run in the background; only measurements may, and a measurement job must write to its own
log file and nothing else.


**The merge discrimination: is the birth census real, or twins?**  The reseeding census calls a
solution "unoccupied" when it is further than ``--merge`` from every carried configuration, so
running it at the study's dedupe convention and at a loose 1e-2 separates genuine new solutions from
census twins (deterministic given the seed, so the two runs agree):

| merge | samples with unoccupied solutions | total | contiguous runs |
|---|---|---|---|
| 1e-3 (default) | 45 of 168 | 253 | - |
| 1e-2 | 45 of 168 | 253 | 1 (123, 167) |

数量几乎不变（45 -> 45）⇒ 那 253 个解**不是**被跟踪分支的孪生，而是这些采样纤维里货真价实、没有任何轨迹跟随的实解——**Q20 的盲区被独立证实**。  Command: ``python3 -m study.exp50_births_and_stops --pose-seed 0 --merge 1e-2``.


### 3.38 The loop's event timeline in one place: deaths early, births late, and the sheet change between

The three measurements on the same object (pose seed 0, pair (0, 7), 168-sample loop, command
``python3 -m study.exp50_births_and_stops --pose-seed 0``) belong to one timeline, and reading them
apart hides the structure:

| sample range | what is measured there |
|---|---|
| 31 | three branches die (tracks 1, 7, 9) -- classified folds by the next-target census; tracks 7 and 9 die as a pair (nearest 1.20e-13 rad apart just before) |
| 45 | three more die (tracks 2, 5, 8) -- folds |
| 1-30 (30 samples) | pairwise gap below 1e-2: the approach phase of the pairs that die at 31 and 45 |
| **123-167** | **the carried branches no longer cover the fibre**: 45 samples with unoccupied solutions, 253 in total, one contiguous run to the end of the loop |

So the loop's discriminant crossings are **not** symmetric in kind: the deaths happen in the first
quarter (two clusters, at 31 and 45), while from sample 123 to the end solutions exist that no track
follows -- the birth side of the same phenomenon.  A detector built on "two live tracks collide and
die" registers the first two clusters and is structurally blind to the third event, which is exactly
the asymmetry Q20 records and why the seeds-1/2 runs reported zero collisions while their tracked
pairs still changed sheet.

What this timeline makes obvious about the next step: the births must be **picked up at the run's
start (sample 123) and carried forward** (Q23's remainder).  Then the same table has births as well
as deaths, the live count stops being monotone, and the crossing count of a loop becomes a property
of the loop rather than of which pair happened to be tracked.

Instrument note: the run that produced this table is deterministic in its seed and parameters, and
two delayed jobs of this study wrote to its log file at different times; their outputs agree, which
is what makes the table usable -- but the rule stands that only measurement jobs may run in the
background and only into their own log (3.37's bookkeeping hazard).



**Seed-1 rerun inside the reseeding build: the verdict reproduces, the census did not finish.**  The
same pose seed 1 that 3.37 used, now run with the per-sample reseeding phase compiled in
(``exp50 --pose-seed 1``), reproduces the stop verdict exactly -- **3 fold(s), 0 tracker drop(s)** --
and its log
(``.scratch/exp50_birth_s1.log``) stops after that line: the reseeding census (103 samples x 120
restarts) had not printed when this was written, so the *birth* half of the seed-1 comparison is
**未决**, with the log path recorded so the next session can read it rather than re-run it.  What is
established is that adding the census phase did not perturb the stop classification: the two
independent seed-1 runs (before and after the phase was added) agree on 3 folds / 0 drops.


### 3.39 Births come in several events per loop, not one: the two-pose census

The seed-1 census finished on a rerun with fewer restarts (``--sample-seeds 40``; the first attempt's
process died before printing, which is recorded as the pending item it was):

| pose seed | loop samples | lift | stop verdict | samples with unoccupied solutions | total | contiguous runs |
|---|---|---|---|---|---|---|
| 0 | 168 | 4/10 arrivals | 6 folds / 0 drops | **45** | 253 | **1**: (123, 167) |
| 1 | 103 | 5/8 arrivals | 3 folds / 0 drops | **43** | 126 | **3**: (36, 39), (42, 42), (65, 102) |

Two things this settles and one it opens:

* The phenomenon is **generic**, not a seed-0 accident: on both poses a large fraction of the loop's
  samples (45/168 and 43/103, i.e. 27% and 42%) carry real solutions that no carried track occupies.
* The *structure* differs: seed 0's is one run to the end of the loop, seed 1's is **three runs**
  (a short one at 36-39, a singleton at 42, and a long one from 65 to the end).  So a loop has
  **several birth events**, not one -- and Q23's remainder (pick the births up and carry them
  forward) has to handle a *sequence* of them, re-running the pickup at every sample where the live
  set fails to cover the fibre, rather than once at a single "birth sample".
* Opened: whether the run boundaries (36, 42, 65) coincide with the loop's *deaths* (which for seed 1
  sit at samples 51-52 from 3.37) or with crossings the gap detector misses.  That is a per-sample
  comparison the next instrument should print directly: deaths, births and gap minima in one table.

Control unchanged: the degenerate loop is 8/8 arrivals with no permutations, collisions or stops.


**Do the birth runs coincide with the deaths?  Measured: no, on both poses.**  The death samples are
known from the same runs (3.37: seed 0 dies at 31 and 45; seed 1 at 51 and 52), and the birth runs are
(123, 167) for seed 0 and (36, 39), (42, 42), (65, 102) for seed 1:

| pose | death samples | birth runs | relation |
|---|---|---|---|
| 0 | 31, 45 | (123, 167) | births begin **78 samples after** the last death |
| 1 | 51, 52 | (36, 39), (42, 42), (65, 102) | one run **before** the deaths, one **between** them and the long run **after** |

So the two event kinds are not two views of the same samples: deaths are where a *tracked* pair
coalesces, births are where the live set stops covering the fibre, and on this loop they sit in
different places (seed 0: deaths early, births late; seed 1: a birth before the deaths, a singleton
between them, and the long run after).  That also explains why a gap-threshold detector reports
nothing for seed 1: the samples where its deaths happen carry gaps of order 7e-2 (3.37), i.e. the
dying branches there are the tracked ones and their partners are not carried at all -- the events
that matter are the ones the live set cannot see.

The next instrument therefore has to print the three families in one table (deaths, births, gap
minima per sample) rather than comparing them across notes, which is what this entry had to do.


**Is the 3R's workspace split grid-dependent?  Checked at twice the resolution.**  The three-region
claim of 3.32 was measured on a 25x25 grid of tool points; at 40x40 (2.6x the
samples per axis) it reads:

| grid | 4 solutions (inner) | 2 solutions (outer) | unreachable |
|---|---|---|---|
| 25x25 | 110 (18%) | 310 (50%) | 205 (33%) |
| **40x40** | **272 (17%)** | **823 (51%)** | **505 (32%)** |

The same run reproduces phase 1 and 2 exactly (``rank J = 3`` on 200/200 with worst ``sigma_min``
5.16e-04; **2 aspects** at 300x300 with cell counts 44700/44700 and 0.67% singular cells; the fibre
histogram 25 poses of 2 solutions with multiplicities (1,1) and 15 poses of 4 with (2,2)), and the
quartic-coefficient agreement across azimuths is 1.18e-13.  So the workspace
split is a property of the arm at this resolution, not a grid artefact -- which is what the aspect
labelling had already shown at three resolutions (3.32).

Also recorded: a flood fill that precomputes its list of start cells labels 3480 components on a
two-component torus (the list is computed before any labelling, so every free cell starts a new
component). Cheap to fix, invisible without a sanity check on the component count -- which is why
the run prints the component count and the cell-count balance (44700/44700) rather than just the
label field.

### 3.40 Carrying the births forward closes the loop's real event table -- and two tracker faults fall out of it

Q23's remainder, implemented.  A forward lift can only follow what it carries, so the per-sample
census of 3.37 could *point at* the blind spot but not repair it.  The tracker now has an ``admit``
hook (``study/taskspace.py``): at every sample the caller may hand it configurations and they become
ordinary tracks (``Lift.born`` records the sample), so a pair that becomes real mid-path is followed
to the end like anything else.  ``study/exp50_births_and_stops.py`` drives it with a ``FiberAdmitter``
that (a) seeds a census from the previous sample's solutions plus fresh restarts, (b) admits every
verified real solution no live track covers, and (c) classifies it **structurally**: continue it
*backwards* one sample with the same residual convention (``homotopy.correct``,
``se3_error(FK(q), target)``).  A corrector that reaches the previous target without a basin jump
means the branch was already real there (a dropped track, or one the earlier census missed); a
corrector that cannot get there has run into the discriminant *between* the samples, i.e. a **birth**.
Two further additions read the result: ``root_census`` (fresh multi-start seeds, every root
Newton-polished, then deduplicated -- ``census``'s own tolerance cannot be tightened, at 1e-6 it
chains one root into 83-109 "solutions"), and a fold-pair test (continue the *next* sample's fibre
back one sample and see which live roots have no continuation).

**Run 1, the unchanged instrument** (``python3 -m study.exp50_births_and_stops --pose-seed 0``,
7m0s): the raw reseeding phase still reads exactly 45 of 168 samples and 253 unoccupied solutions in
the single run (123, 167) -- 3.37's numbers reproduce bit for bit.  The carried lift then reads:

* live count is no longer monotone: **10 -> 7 -> 4 -> 6 -> 10 -> 12**, in runs 1-31: 10, 32-45: 7,
  46-122: 4, 123-127: 6, 128-165: 10, 166-167: 12;
* admissions at **3 samples**: 123 (+2), 128 (+4), 166 (+2); all 8 classified as **births**
  (backward residual 5.34e-16 .. 3.22e-02, against the 1e-9 tolerance), **0** dropped branches
  re-admitted, **0** samples with an odd birth count (real folds make solutions in pairs, so an odd
  count would mean the census found only one of the two);
* control (degenerate loop, hook armed): 20 samples, 10 -> 10 live, **0 admissions, 0 births,
  0 collisions**.

**Two tracker faults fall out of the comparison with the fibre.**  The run also prints, for every
event sample, an independent census of the fibre, and against it the carried tracker is wrong:

| sample | fibre (census) | live tracks (carried) | live *distinct* | signs (census) | outside the joint box |
|---|---|---|---|---|---|
| 0 | 10 | 10 | 10 | 5/5 | 0 |
| 31 | **8** | 10 | 9 | 5/4 | 1 |
| 32 | 6 | 7 | 7 | 4/3 | 1 |
| 45 | 6 | 7 | 7 | 4/3 | 1 |
| 46 | **2** | 4 | 4 | 2/2 | 2 |
| 123 | 4 | 6 | 6 | 3/3 | 2 |
| 167 | 10 | 12 | 12 | 6/6 | 2 |

* **Fault 11 -- a fold jump leaves a duplicate track.**  The fault report localises it exactly:
  ``merged samples by pair (first..last: tracks) [(2, 31, [7, 9])]``.  Tracks 7 and 9 are 2.56e-09
  apart at sample 2 (they were >= 3.96e-01 apart at sample 1 and every base pair is >= 7.07e-01
  apart) and ~1e-14 apart for the next 28 samples, then both stop at sample 31.  A *genuine* fold
  coalescence is instantaneous -- the pair coincides at one sample and both are gone at the next --
  so a merge that **persists** is a tracker fault: the corrector crossed the fold between samples 1
  and 2, failed to die, and settled on the *other* real solution, leaving two tracks on one branch.
  Root cause: the jump guard measures the distance between the corrected configuration and the
  *tangent prediction*, and near the fold the prediction itself is pushed onto the neighbouring
  branch, so a jump of more than 0.4 rad passes a 0.3 rad guard.  Fix: read the merged pair and
  retire the newer track (two distinct branches cannot coincide away from the discriminant) --
  ``prune_merged``.
* **Fault 12 -- the tracker never checked the joint box.**  Configurations it reports are solutions
  of the pose but not necessarily configurations the arm can be in: at full strength 2 of the 12
  final live tracks are outside the limits (`--box-guard` off), and the distance to the nearest base
  solution reaches 2.46e+02 rad for such a track (the census, by contrast, is clipped to the box by
  ``solve_lm``).  Fix: the ``feasible`` predicate, checked wherever the corrector's clearance is
  checked; exp50 passes "inside the box, 1e-6 of slack".  Both fixes are **off by default** in the
  run so that the previously recorded Q20/Q21 numbers stay reproducible with the instrument that
  made them.

**Run 2, repaired** (``python3 -m study.exp50_births_and_stops --pose-seed 0 --box-guard
--prune-merged``, 7m1s).  The box guard alone removes the jump (``collisions 0, merged 0``, the
duplicate list is empty; pruning was armed and had nothing to do), and the event table becomes
*verified*: the carried live set matches the independent polished census at **all 13 probe samples**
(10, 10, 8, 8, 6, 6, 2, 4, 4, 8, 8, 10, 10), the live runs become 1-31: 8, 32-45: 6, 46-122: 2,
123-127: 4, 128-165: 8, 166-167: 10, and every fold kills **exactly two roots of opposite ``det J``
sign**:

| between samples | fibre | dead roots (holding track, sign, ``sigma_min``) |
|---|---|---|
| 1 and 2 | 10 -> 8 | (3, -1, 4.4e-03), (9, +1, 3.2e-03) |
| 31 and 32 | 8 -> 6 | (1, +1, 1.2e-02), (7, -1, 1.2e-02) |
| 45 and 46 | 6 -> 2 | (2, -1, 1.4e-02), (4, -1, 1.2e-02), (5, +1, 1.5e-02), (8, +1, 1.2e-02) |

So the loop's **real event table** is three death events (-2, -2, -4) and three birth events
(+2, +4, +2): **8 branches die and 8 are born**, the fibre size returns to its starting value
(10 -> 10, the witness path's last configuration being 3.4e-06 from the first pose), and at the end
the 10 live tracks sit on the 10 base solutions (largest distance to the nearest base solution
1.95e-05 rad).  The two folds inside the 45-46 interval show up as four dead roots paired by
opposite sign, i.e. two events between two samples -- which is what the resolution question (Q24)
is about.  Note what the births *are*, structurally: after the loop the live set is the fibre again,
so the 8 born tracks replace exactly the 8 that died.  On the real side the loop's lift is therefore
a **partial** map on the fibre -- only 2 of the 10 starting postures (tracks 0 and 6) can be carried
around it, ending on solutions 7 and 6 -- and the births are what closes the image back onto the
fibre.  That is the real-side counterpart of the complex monodromy being a total permutation.

**Consequence for the earlier crossing readings -- flagged, not yet re-measured.**  The signature
"two tracks merge to ~1e-14 and then die together" was read in 3.34/3.38 (exp49's 118-sample loop,
3 collision samples 46/47/48, "a pair merging to 9.12e-15 then dying together") as a discriminant
crossing.  Fault 11 shows that the *same* signature is produced by a fold jump, and the
discriminator is duration: a genuine coalescence is one sample, a jump persists.  3.34's merge
lasted three samples, so that number must be re-measured with the census/fold-pair instrument before
it is quoted as a crossing; what survives unchanged is the death side of the A->B answer (the stop
samples are verified folds by the local census in 3.37 and by the fibre table above).  Registered as
Q24 together with the resolution experiment: subdivide the witness polyline around a multi-event
interval (the -4 at 45-46) and check that the folds separate into individual crossings, each with
the pair's gap going to zero and ``sigma_min`` to zero at the crossing pose.

Commands: ``python3 -m study.exp50_births_and_stops --pose-seed 0`` (raw + carried, unchanged
instrument) and ``... --pose-seed 0 --box-guard --prune-merged`` (verified table).  Logs:
``.scratch/exp50_carry_s0b.log``, ``.scratch/exp50_guarded_s0.log``.  Also fixed while here: the live
run report indexed ``counts`` from 0, so the printed sample ranges were shifted by one (a display
fault only; the sample numbers in this entry are the corrected ones).


### 3.41 The -4 event is two folds, and each one is located in task space (Q24)

``python3 -m study.exp51_event_resolution --pose-seed 0 --subdivisions 96`` (48 s; new module,
number registered in ``QUESTIONS.md`` before use).  Two questions were open on the loop's event
table: does the interval that changes the fibre by four hide two folds or one degenerate event, and
where *in task space* does a crossing happen?  Both are answered by refining the path rather than the
detector: the witness polyline is subdivided in joint space (``path[k] + t (path[k+1] - path[k])`` for
96 values of ``t``, the ``t = 0`` sample kept so the starts are attributed to the right target) and
the same guarded/pruned tracker walks it.

**The -4 resolves.**  Interval 45-46 gives **two** changes instead of one -- sub-sample 37: 6 -> 4
(dead roots, tracks 2 and 5), sub-sample 38: 4 -> 2 (tracks 1 and 3) -- with the census agreeing at
each step (4 then 2) and the two pairs of opposite ``det J`` sign.  The single folds at 1-2 and
31-32 stay single at this resolution (one change each, a pair of opposite signs).  So the loop has
**six** fold crossings, not five, and the sample-interval count was hiding one of them.

**Each crossing is bracketed in task space.**  ``localise_crossing`` bisects the *pose* interval
(``ts.se3_interpolate``, so every bisection point is a pose) between "the pair is still two distinct
real roots" (each seed continued locally, with a jump guard, so a census cannot masquerade as a
continuation) and "it is not", to 1e-6 of the event interval:

| event (samples) | tracks | bracket | pair gap at the bracket | coalesced ``sigma_min`` | exponent | ``tau*`` from ``gap^2`` line |
|---|---|---|---|---|---|---|
| 1 -> 2 | 3 / 9 | [0.791977, 0.791978] | 3.03e-04 | 2.78e-06 | 0.503 | 0.791873 |
| 31 -> 32 | 1 / 6 | [0.149731, 0.149732] | 4.55e-06 | 2.19e-08 | 0.503 | 0.149693 |
| 45 -> 46 (first) | 2 / 5 | [0.405725, 0.405726] | 7.55e-06 | 2.00e-07 | 0.501 | 0.405701 |
| 45 -> 46 (second) | 1 / 3 | [0.379368, 0.379369] | 1.94e-06 | 6.17e-08 | 0.501 | 0.379344 |

Two things are certified there and neither is assumed.  First, **both numbers go to zero at the
crossing**: the pair's gap is 1.9e-06 to 3.0e-04 rad and the coalesced configuration's clearance is
6.2e-08 to 2.8e-06 -- three to five orders below the ``sigma_min ~ 1.2e-02`` measured at every stop
(3.37), which is exactly the difference between "the samples straddle a fold" and "the pose is on the
discriminant image".  Second, the gap vanishes as the **square root of the task offset**: a geometric
ladder towards the crossing fits exponents **0.501, 0.501, 0.503, 0.503** -- the task-path-side
counterpart of the split exponent measured on the DH side (exp27: 0.389-0.448 for three folds), and
the reason the ladder is not decoration: fitting ``gap^2`` as a straight line in the offset gives an
independent estimator of the crossing (``tau*`` in the last column) which agrees with the bisection
bracket to 1e-4 to 2e-5 of an event interval, i.e. two different instruments, one geometric and one
algebraic, locating the same pose.  Every pair's two branches carry **opposite det J signs**
(+-1.0e-04 to +-1.6e-04) at the bracket, i.e. the crossing is between the two aspects the fold
separates (Wenger).

**One fold is certified as a singular configuration of the pose it crosses.**  ``refine_fold_pose``
solves the augmented system of the *task* map -- unknowns ``(q, tau)``, equations ``fk(q) = T(tau)``
(six) and ``det J(q) = 0`` (one), with a weight-1e-3 row pinning ``tau`` to the event segment (without
it the Newton walks off along the extrapolated pose line and "converges" to a fold of a pose the path
never visits: measured ``tau = -90``, ``5340``, ``-15696`` with small residuals).  For the 45-46
second fold it converges to ``tau = 0.379368``, residual **4.3e-10**, ``det J = 1.7e-17``,
``sigma_min = 1.7e-16``, and ``tau`` equals the bracket's ``lo`` to all six digits printed -- a point
that is simultaneously a solution of the pose and singular.  For the other three folds this refinement
stalls from a bracket-level seed (residuals 4e-3 to 2e-2, ``det J`` 4e-3 to 2e-2), which is reported
rather than papered over: those three crossings rest on the bracket plus the two zero certificates plus
the exponent, not on the augmented point.  The seed that does work is the midpoint of *the bracket's
own roots*, which is why ``localise_crossing`` returns them.

**Control.**  Subdividing a degenerate (constant) path gives 6 -> 6 live, 0 collisions, 0 merges,
0 stops -- the events are the arm's, not the subdivision's.

What is left of Q24 is the other half: the older exp49 reading ("three collision samples, a pair
merging to 9.12e-15 then dying together" on the 118-sample loop) still needs the same treatment --
``merged`` duration plus a bracket -- before it can be quoted as a crossing.  Registered, not done.


### 3.42 The A->B answer with located ends: each posture's feasible arc, and where it ends

``python3 -m study.exp52_feasible_arcs --pose-seed 0`` (new module, number registered first; about
20 s).  exp50 answered "which posture can track how much of the loop" in samples, and exp51 located
the crossings; this module puts the two together in the form the planning question is asked: **one
loop, run once per posture, with the arc measured as a fraction of the loop's Cartesian length and
its end named**.  Track indices equal base-solution indices here because nothing is admitted
mid-path, so the table speaks about *named* postures.

The lift is the repaired instrument (box guard + duplicate pruning) with no admission: live
10 -> 2, 8 stops, **0 collisions, 0 merges, 0 duplicates** -- the faults of 3.40 are gone.  The loop's
Cartesian length (sum of ``se3_error`` steps) is 4.5335 over 167 intervals.

| posture | sign | arc (samples) | Cartesian fraction | ends at |
|---|---|---|---|---|
| 0 | -1 | 0..167 | **100.0%** | whole loop |
| 6 | +1 | 0..167 | **100.0%** | whole loop |
| 1 | +1 | 0..31 | 23.2% | crossing at sample 31.2693 (`sigma_min` 1.3e-07), paired with 7 |
| 7 | -1 | 0..31 | 23.2% | crossing at 31.2693, paired with 1 |
| 2 | -1 | 0..45 | 30.8% | crossing at 45.3948 (`sigma_min` 8.2e-08), paired with 5 |
| 5 | +1 | 0..45 | 30.8% | crossing at 45.3948, paired with 2 |
| 4 | -1 | 0..45 | 30.8% | crossing at 45.3847 (`sigma_min` 2.7e-07), paired with 8 |
| 8 | +1 | 0..45 | 30.8% | crossing at 45.3847, paired with 4 |
| 3 | -1 | 0..1 | **0.9%** | crossing at 1.4176 (`sigma_min` 1.8e-06), paired with 9 |
| 9 | +1 | 0..1 | **0.9%** | crossing at 1.4176, paired with 3 |

Every arc ends at a located crossing, every pair is one ``det J`` sign against the other, and the
pairs are exactly the four the fold-pair test finds (exp51): (3, 9) at 1.4176, (1, 7) at 31.2693,
(4, 8) at 45.3847 and (2, 5) at 45.3948 -- the last two being the resolved halves of the -4, 0.010
samples apart.  So the operational form of "the feasible-path region is the image of a uniqueness
domain" is: **a posture owns one arc of this loop, its length is the part of the path whose image
stays inside that domain's image, and the arc ends exactly where the path crosses the discriminant
image**.  Two postures own the whole loop and eight do not; that is what "同分支 != 同可行路径域"
means in numbers on this path.

**How accurately is a crossing located?  Two routes, and the honest answer is 0.005 of a sample.**
exp52 seeds its bisection with a *march* (the pair's two configurations walked forward in 64
substeps with the same corrector, stopping where they stop being two roots), because seeding the
bisection a whole interval away fails early and biases the bracket late (measured: 45.3847/45.3948
from interval-level seeds against 45.3792/45.3894 from sub-interval-level seeds -- the same two folds
named 0.005 samples apart).  With the march, the positions are **stable to six decimals across march
resolutions 32/64/128** (1.417569, 31.269323, 45.384651, 45.394843), and they still differ from
exp51's tracker-based brackets by 0.003 to 0.0055 samples.  The reason is structural rather than
numerical: a *tracker* can give up slightly before the pair ceases to be real (near the fold the
corrector is ill-conditioned and the step guard bites), so a bracket built on "the last sample the
tracker reached" can end before the true crossing, while a bracket built on "the seeds still correct
to two distinct roots" can reach further.  Both routes agree on the identity of the pairs, on the
resolution of the -4 into two folds, on their order, on the exponents (0.499-0.510) and on the two
zero certificates; where they disagree, by 0.005 of a sample interval -- about 0.1 mm of Cartesian
travel on this loop -- the difference is **未决** and is reported as the accuracy of the localisation.
The lesson generalises the one from 3.33: a bisection bracket is 1e-6 wide because that is the
bisection's stopping rule, **not** because the crossing is known to 1e-6.

The augmented system (exp51's ``refine_fold_pose``, seeded from the march) certifies one fold on this
route too: pair (2, 5) at sample 45.394813, residual 3.6e-05, ``det J = -7.8e-13``,
``sigma_min = 1.0e-11``; the other three stall (residuals 4.5e-03 to 1.5e-02), as in 3.41.

Control: the degenerate loop gives 10 -> 10 live, 0 stops, 0 collisions, 0 merges.

Method note, recorded because it produced a wrong table for one run: the subdivided run numbers its
tracks by position in its start list, and pruning can retire one, so a group's *holders* must be
mapped through the start list to recover the original posture labels -- reading the group index as a
posture index named posture 6 as the partner of 1 at the 31-32 fold, where the whole-loop run and
3.40's table both say 7.  exp51 now prints both (subdivided-run track and original postures).


### 3.43 The 118-sample loop re-measured: the "three collision samples" were a jump artefact, and the loop shows a drop and a birth instead

The last piece of Q24: 3.34 read exp49's loop as crossing the discriminant at three collision samples
(46/47/48, a pair merging to 9.12e-15 and dying together), and 3.40 showed that the same signature --
a merge that *persists* -- is what a fold jump produces.  exp49 now takes the two repairs of 3.40
(``--box-guard``, ``--prune-merged``) and a phase 5 (``--locate``) that brackets crossings with
exp51's instrument instead of reading them off a gap threshold.  Both runs:
``python3 -m study.exp49_sr5_uniqueness --pose-seed 0`` (unchanged instrument, logs
``.scratch/exp49_asis_s0.log``) and ``... --box-guard --prune-merged --locate``
(``.scratch/exp49_repaired_s0.log``).

**The unchanged run reproduces the recorded numbers exactly**, and names the fault: arrivals 2/10,
shortest gap 9.12e-15, 3 collision samples, **3 merged samples, one run, samples 46-48, by tracks
[0, 7]** -- a three-sample merge, i.e. by 3.40's discriminator a tracker fault and not a
coalescence.

**With the repairs the collision signature is gone**: arrivals 1/10, live 8 -> 1, shortest gap
**2.20e-01** (22 times the 1e-2 collision threshold), **0 collisions, 0 merged, 0 duplicates**.  The
posture change survives, and this is the point of 3.34: **track 1 still arrives at solution 6 with
joint-space distance 0.0e+00** -- a prescribed Cartesian loop that changes posture without touching a
singularity -- and the degenerate control is still 10/10 with 0 collisions.  So the corrected
statement is: the loop changes posture, and its crossings are **invisible to the collision detector**
(the Q20 blind spot), not "detected at samples 46-48".

**Phase 5 locates two crossings on this loop** (bracketed in task space, 64-substep march, 40
bisections):

| interval | tracks (signs) | crossing (sample) | pair gap | coalesced ``sigma_min`` | exponent | ``gap^2`` zero |
|---|---|---|---|---|---|---|
| 0 -> 1 | [3] / [9] (-1 / +1) | 0.342537 | 5.62e-04 | 5.86e-06 | 0.500 | 0.342537 |
| 41 -> 42 | [2] / [5] (-1 / +1) | 41.288747 | 2.80e-06 | 1.26e-07 | 0.501 | 41.288747 |

Same signatures as 3.41/3.42: opposite ``det J`` signs, two zero certificates, square-root vanishing,
and the two independent position estimates agreeing to six decimals.  The augmented refinement does
not converge here (residuals 9.5e-04 and 2.5e-01), which is reported as such.

**What the fibre says about the other stops -- and it is neither "fold" nor "nothing".**  Every
remaining stop was checked the same way (census at the next sample, continued back one sample):

* **34 -> 35**: the tracker's track 7 has no continuation in the next fibre, yet that census finds
  **8** solutions -- the same number the tracker held.  One root of the old fibre therefore vanished
  *and* at least one new one appeared inside one sample interval: a death and a birth that cancel in
  the count.  A detector built on "the live count drops" (or on gaps) can see neither.  Census
  completeness at that sample is the usual caveat; recorded as 未决 in detail, measured in kind.
* 43 -> 44 and 45 -> 46: the fibre **grows** (4 -> 6 and 3 -> 4 roots), i.e. births, while single
  tracks stop -- again invisible to a death-based detector.
* 48 -> 49: census 2 for 2 live tracks, with one dead root: a birth replacing the death.
* 41 -> 42: the fibre drops by four (two folds); the tracker sees one pair ([2]/[5], located) and one
  unpaired root [4] whose partner is the root left uncovered when track 7 stopped earlier -- which is
  why an unpaired death is not automatically a tracker fault.

So this 118-sample loop, which 3.34 read as "one crossing event", actually exhibits **two located
crossings, one tracker drop, and at least two births**, and the collision counter that was used to
find the crossing reads zero once the artefact is removed.  That is the strongest form of Q20's
lesson so far: on a loop that demonstrably changes posture, the forward/collision instrument reports
nothing at all.


### 3.44 A second Pieper neighbour: axes 2 || 3 || 4, measured, and it seeds the poses SR0 cannot

Q25.  SR0 closes the *wrist* (zeroing link 5's y-offset makes axes 4, 5, 6 concurrent -- one of
Pieper's two sufficient conditions) and leaves a seed gap: on some poses its position sub-problem has
no real solution at all, so the continuation cannot start (3.x, exp13/exp43).  Pieper's *other*
condition is the parallel one -- three consecutive axes parallel -- and SR5 already has axes 2 || 3,
so a single twist change is enough: set link 4's rotation from ``Rx(-90 deg)`` to ``Rx(0)`` or
``Rx(180 deg)`` and axes 2, 3, 4 are parallel.  ``python3 -m study.exp53_second_neighbour --poses 60
--seeds 200 --homotopy --reference-seeds 400`` (10m31s; log ``.scratch/exp53_full.log``).

**Structural certificates** (pairwise axis fingerprint; adjacent pairs are configuration-independent,
and the rank test is 200 random configurations):

| arm | angle(3,4) | dist(3,4) | wrist gap (4,6) | full rank | worst ``sigma_min`` |
|---|---|---|---|---|---|
| SR5 (unchanged) | 90.000 deg | 0.0500 m | 0.1360 m | 200/200 | 1.4e-04 |
| SR0 (wrist closed) | 90.000 deg | 0.0500 m | **0.0000 m** | 200/200 | 1.8e-03 |
| parallel, link 4 twist 0 deg | **0.000 deg** | 0.4031 m | 0.1360 m | 200/200 | 1.1e-04 |
| parallel, link 4 twist 180 deg | **0.000 deg** | 0.4031 m | 0.1360 m | 200/200 | 3.4e-04 |

So the parallel candidates satisfy Pieper's parallel condition *and* are different arms (the wrist
offset of 0.136 m is untouched), and they are not degenerate (full rank everywhere sampled).

**Coverage of SR0's seed gap** (60 poses drawn as ``fk`` of random in-limit configurations, so from
the arm's own workspace; SR0 by its closed form ``study.sr0.solve``, the others by a 200-seed
``solve_lm`` census):

| candidate | both solve | SR0 only | **candidate only (covers the gap)** | neither |
|---|---|---|---|---|
| SR5 (census control) | 53 | 0 | 7 | 0 |
| SR0 (reference) | 53 | 0 | 0 | 7 |
| parallel, twist 0 deg | 53 | 0 | **7** | 0 |
| parallel, twist 180 deg | 53 | 0 | **7** | 0 |

SR0 is blind on **7 of 60 (12%)** here; the recorded 20-30% came from other samples of the same
``fk`` construction (exp43: 4/20, 6/20), and with 7/60 the binomial interval overlaps it, so the two
are not in conflict -- the point of this table is the *column*, not the rate.  Both parallel arms have
real solutions on **all seven** blind poses, and the unmodified-SR5 row certifies the census does not
under-detect (it is blind nowhere SR0 solves).

**Can the parallel arm actually seed SR5?**  ``homotopy.track`` along link 4's twist
0 deg -> -90 deg (SR5's own twist, read off the matrix), one track per parallel-arm solution, matched
against a 400-seed SR5 census at the same pose:

| blind pose | SR5 reference | parallel starts | arrived | reference solutions reached |
|---|---|---|---|---|
| 1 | 8 | 4 | 4 | 4 |
| 2 | 2 | 4 | 1 | 1 |
| 3 | 8 | 4 | 4 | 4 |
| 4 | 8 | 4 | 4 | 4 |
| 5 | 4 | 4 | 1 | 1 |
| 6 | 3 | 4 | 0 | 0 |
| 7 | 2 | 4 | 2 | **2 (complete)** |

Totals: **1 of 7 blind poses fully recovered, 16 of 35 reference solutions reached from 16 arrivals
of 28 starts**; 12 starts die on the way (the twist path crosses folds).  So the second neighbour
*does* remove the "no seed at all" obstruction -- 7/7 of the poses SR0 cannot start on now have real
solutions to start from -- but a straight twist path is not yet a complete solver: it recovers about
half of SR5's solutions.  Registered next steps: curved paths in the parameter space (exp13's trick),
the 180 deg variant as a second start set, combining SR0's seeds with the parallel ones, and writing
the closed form for the parallel family (Pieper's parallel condition is a theorem, but a solver
should exist before this is called a *closed-form* neighbour -- 未决).

**Instrument fault found on the way (symptom / root cause / fix).**  The first version of the
homotopy path interpolated link 4's twist to ``+90 deg`` -- the sign of the *stored* matrix -- while
SR5's link 4 is ``Rx(-90 deg)``.  Symptom: all tracks "arrived" yet **0 of 14 endpoints matched any
SR5 reference solution** at the same pose; a coincidence of two wrong numbers would have been needed
to hide it, and the per-pose line showed arrived 2-4 with reached 0.  Root cause: the path's end was
not the target arm.  Fix: read the twist off the target matrix (``arctan2(R[2,1], R[1,1])``) instead
of writing the angle by hand; after the fix the same poses give 4/4 and 2/2 reached.  The lesson is
the same as 3.x's: an endpoint must be *checked against the object it claims to be*, not assumed from
the construction.


**Neither curved paths nor the second arm improve the *distinct* coverage -- and the reason is
structural.**  ``python3 -m study.exp53_second_neighbour --poses 60 --seeds 200 --homotopy
--reference-seeds 400 --retries 3 --second-arm`` (11m18s, log ``.scratch/exp53_curved.log``): 56
starts (both parallel arms), **103 curved retries** along bumps in the 18 link translations that
vanish at both ends (exp13's trick, moved to the twist path), **24 arrivals** -- against 16 arrivals
from 28 straight starts -- and still **16 of 34 distinct reference solutions reached, 1 of 7 poses
complete**.  Arrivals rose while distinct coverage did not: the extra arrivals land on solutions
already reached.  The reason is visible in the census counts: at these poses the parallel arm has
**4 real solutions** where SR5 has 8 (and 2-4 where SR5 has 4), so the remaining SR5 solutions have
*no counterpart at* ``s = 0`` -- they must be **born** along the twist path, and a forward tracker
cannot create a branch it does not carry.  That is Q23's lesson in parameter space instead of task
space: the fix is to admit births along the DH homotopy (census each ``s`` sample and add the
uncovered solutions), not to look for a better path.  Registered as the next step for Q25.

Also recorded: the *reference* set is a sample, not a set -- with the rng stream consumed differently
(``--second-arm`` doubled the start censuses), one pose's 400-seed reference changed from 3 solutions
to 2, which is why the totals here read 34 where 3.44 read 35.  A reference that moves when unrelated
code draws more random numbers is an instrument fragility worth remembering: the honest reading is
"16 of about 34".


### 3.45 The franka component count of 3.30 does not reproduce on today's model -- measured, unresolved

Starting ④ (Q9: turn the lower bound 13 into a count) turned up a consistency problem that has to be
recorded before any count is claimed.  ``python3 -m study.exp18_franka_redundant --robot franka
--seeds 400`` (today, with the step budget made a flag, ``--steps``) gives:

* **67 solutions** at seed 0's pose, not the 26 of 发现 3/3.30; the smallest pairwise gap between
  them is **8.83e-03** rad and the clearances run 0.071 to 0.215, so they are not census twins -- the
  fibre really is that large on today's model (``ModelFactory.create("franka", backend="casadi")``,
  joint spans 6.109/4.712/5.411/6.109/6.109/6.109/6.109 rad);
* **25 self-motion components** traced with a 4000-step budget, of which **8 never close and never hit
  a joint limit** -- they run the full 4000 steps (~200 rad) and are truncated by the budget, so 25 is
  an *upper* bound, not a count; with the recorded 600-step budget the same runs are more truncated
  still.  The instrument now says so itself (the end-reason census prints "NOT a count" when any trace
  is truncated);
* the cross-component chamber walk connects 5 of 6 sampled pairs in ``Q\\Sigma``, as recorded.

So either the franka model or the census changed since 发现 3 was written, or that record was made on a
different pose; the two cannot both describe today's model, and no component count should be quoted
until the fibre census, the joint limits and the trace budget are re-measured together.  Registered on
Q9; the useful part of this round's probe is that the counting instrument now *refuses* to report a
number it has not earned.



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
