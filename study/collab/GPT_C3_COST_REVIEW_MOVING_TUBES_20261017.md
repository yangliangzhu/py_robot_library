# C3 cost review: retain the stop, reopen the mathematical representation

## What the report establishes

Reviewed `T4-gate-c3a.md` and the fixed-center certificate/planner structure in
`exp72_gate_c3a_edge.py`. No long certification task was launched and the 4529
root-13 certificates were not independently reverified in this review.

The available measurements justify not starting C3b with this implementation.
They do not rule out feasible certification in another representation.
The report already says the complete six-root edge has not been certified.

Corrections required when summarizing the report:

* The complete-edge and three-loop costs are extrapolations, not completed-run
  measurements. Planned segment counts for five roots are not accepted counts.
* A window between earliest and latest output-file mtimes excludes the time to
  produce the earliest file. It is an output-arrival window, not a measured
  start-to-finish batch wall time. Recover launch/completion timestamps if making
  the latter claim; distinguish worker elapsed seconds from CPU seconds.
* Large pairwise root separation does not bound inverse Jacobians or root speeds.
  Even f_epsilon(x,t)=epsilon*x*(x-1)-t has roots separated by 1 at t=0,
  while their speeds have magnitude 1/epsilon. This is a logical counterexample,
  not a model for SR5. More SR5-specific information is needed to extrapolate
  costs across the entire paths from their first edge.
* A bound kappa(r) approximately linear in r does not prove sharpness. For
  example, natural interval evaluation of x^2-2*x+1 on [1-r,1+r], 0<r<1,
  has width O(r), whereas the exact range is [0,r^2]. Combining equal monomials
  does not remove this cancellation between distinct powers. Saturating a
  selection threshold proves that that bound limits the planner, not that no
  better bound exists.
* From 35 days to 7 days requires 5x speedup; to 1 day requires 35x.
  A claimed 50--150x requirement needs its own target and overhead assumptions.

## Moving-center tube theorem (realified equations)

Write the fixed-orientation translation equations along a straight parameter
edge, using a real scalar s in [-h,h], as

    f(x,s) = f0(x) - s*b.

Here f0 includes the exact midpoint target, b is the exact realified translation
velocity (possibly from a complex direction), and h > 0 is rational. Choose an
exact rational predictor c(s)=c0+s*v, constant rational preconditioner B, and
positive diagonal D=diag(r_i). Seek x(s)=c(s)+D*y(s), ||y||_infinity <= 1.

The fixed-point map for each s is

    T_s(y) = y - D^-1*B*f(c(s)+D*y,s).

Certify, uniformly on the swept tube and the parameter interval,

    eta_i >= sup_s |(D^-1*B*f(c(s),s))_i|,
    K_ij >= sup_(s,y) |(I-D^-1*B*Df(c(s)+D*y)*D)_ij|.

If max_i sum_j K_ij < 1 and eta_i + sum_j K_ij < 1 for all i, Banach
gives one unique root per slice and a continuous regular lift. The original
comfort policy is preserved by requiring row sums < 1/2 and
eta_i+sum_j K_ij <= 3/4. Check z_j != 0 and trace exclusion on the full swept
tube, just as before. No policy relaxation is involved.

### Rigorous predictor residual bound

Define the vector function q(s)=D^-1*B*f(c0+s*v,s). Then

    q(0)  = D^-1*B*f0(c0),
    q'(0) = D^-1*B*(Df0(c0)*v-b),
    q''(s)= D^-1*B*D^2 f0(c0+s*v)[v,v].

With exact point evaluations for q(0), q'(0), and rigorous componentwise bounds
H_i >= sup_|s|<=h |q''_i(s)|, Taylor's integral remainder gives

    eta_i = |q_i(0)| + h*|q'_i(0)| + (h^2/2)*H_i.

This is a sufficient bound over the whole segment, not endpoint sampling.
Choose v numerically from Df0(c0)*v=b, then freeze its values exactly. The
nonzero predictor defect q'(0) MUST be retained. A rational exact solve is also
possible. Differentiating the existing Laurent system twice gives the required
directional derivatives; precombine with B before interval evaluation if useful.

The fixed-center tube radius must accommodate O(h) root motion. A moving-center
tube needs to accommodate predictor error, ideally O(h^2), instead. This is the
potential advantage, not a guaranteed speedup: the Jacobian bound still ranges
over the center motion as well as the transverse radii. A broad swept rectangle
can lose the same correlations. If that becomes the limiting factor, retain s
as a shared variable using a centered Taylor enclosure before considering a
variable preconditioner or higher-order predictor. Benchmark before expanding.

Do NOT reuse the C2 endpoint-only residual formula: f(c(s),s) is generally
nonlinear in s when c moves.

### Glue and closure

At a common endpoint, the two slice boxes are c_left(h)+D_left[-1,1]^12 and
c_right(-h')+D_right[-1,1]^12. Certify a small endpoint node box N and prove it
lies inside both slice boxes. Uniqueness at that parameter identifies the roots.
Apply the same rule to C1 boxes at the start and at loop closure. The whole swept
bounding rectangle is NOT the endpoint slice box.

## Next bounded experiment, if this stage is resumed

Keep all existing artifacts and the current stop on C3b. Before implementing a
new binary format or committing to a backend rewrite, give one executor a small
comparison task on root 13 near the first archived edge:

1. Reproduce one existing fixed-center certificate as a baseline.
2. Implement the affine predictor bound above, first with equal transverse radii;
   test a short finite ladder of segment lengths with a declared runtime budget.
3. Measure actual accepted length per verification second and bytes, including
   node/gluing cost. Preserve exact proofs and all rejection reasons.
4. Report whether residual, Jacobian enclosure, or exclusion bound is binding.
   Add diagonal scaling only as a separately measured variant.

No promised 50x gain, no full-edge batch, and no group conclusion from this test.
An unfavorable result is useful: it would justify prioritizing backend speed.

For storage, distinguish proof inputs (exact model, path parameters, centers,
radii, preconditioners, coverage/identity data) from derived diagnostics. An
independent checker can recompute the latter; storing every long intermediate
rational is not logically necessary. Omitting them trades verification time
against storage and should be specified explicitly, not called lossy arithmetic.

## Status language

The numerical loops support fixed-arm nonsolvability. C0/C1/C2 are certified
building blocks; C3 is incomplete and currently paused for feasibility work.
Say "numerical evidence; full path certification incomplete". Neither a true
monodromy subgroup inclusion nor fixed-arm nonsolvability has yet been proved
by the numerical permutations alone.
