# C1 review and C2 implementation addendum

## Review scope and result

Reviewed the Laurent system construction, complex rectangular interval arithmetic,
realification, contraction bounds, trace exclusion, and certificate verifier in
`study/exp70_gate_c1_roots.py`. Re-ran:

```
python3 -m study.exp70_gate_c1_roots --verify study/exp70_gate_c1_certificate.json
```

The archived six-root certificate passes (exit 0). No unsound enclosure formula
was found in the reviewed arithmetic. This is source review plus reproduction,
not a second independently implemented interval backend. The certificate proves
six distinct regular roots of the frozen exact model, not completeness of the
fiber or any path permutation.

Using ordinary interval multiplication for a square can give a negative lower
bound when the interval crosses zero. This is conservative, not unsound. In
`ci_recip`, rejection when that conservative denominator bound is nonpositive
may reject a rectangle avoiding the origin; it cannot falsely certify one.

## C1 verifier completeness defect: reproduce and repair

An actual negative test was performed: copy the certificate to
`/tmp/exp70_empty_roots.json`, replace its `roots` object by `{}`, and run the
same verifier on that copy. It prints `PASS: certificate fully re-checked` and
exits 0. The manifest's `tracked_indices` check does not check the certificate's
root entries. Also, a recorded `inconclusive` verdict currently passes its branch
and is omitted from the disjointness checks.

This does not invalidate the six boxes present in the original artifact. It
does invalidate interpreting verifier success alone as a six-root C1 acceptance.

Deterministic repair task for the executor:

1. Require exactly the canonical keys `1,4,6,8,10,13` in full C1 verification.
   Require every verdict to be `certified`; reject unknown verdicts.
2. Validate positive radii and dimensions (12-vector centers, 12 by 12 B).
   Reject malformed records explicitly. If single-root verification is useful,
   expose it as a distinct mode with an explicit required index set and output
   that cannot be confused with full C1 acceptance.
3. Preserve recomputation of all six contraction and all fifteen separation
   checks. The present verifier checks the mathematically sufficient conditions
   kappa < 1 and positive margin; if claiming compliance with the stronger C1
   selection policy, separately check kappa < 1/2 and margin >= r/2.
4. Add negative regressions: empty root set, missing one root, inconclusive root,
   unknown verdict, malformed dimensions. Re-run the unmodified certificate.

No certificate numbers need changing for this repair. Keep the original artifact.

## C2: exact endpoint formula for the entire parameter segment

This simplification applies to the frozen-orientation translation paths in the
Gate C specification. Write the realified equation as

    f(x, lambda) = f_base(x) - realify((lambda*d, 0, 0, 0)),

where the first three complex outputs are position residuals and the last three
are orientation residuals; realification interleaves real and imaginary parts.
For lambda(t) = (1-t)*lambda_A + t*lambda_B and a fixed rational center c and
preconditioner B, put

    v_A = B*f(c, lambda_A),   v_B = B*f(c, lambda_B).
    eta_i = max(abs(v_A[i]), abs(v_B[i])).

All these numbers are exact rationals. Since

    B*f(c, lambda(t)) = (1-t)*v_A + t*v_B,

the maximum absolute value of each component on the WHOLE segment is exactly
eta_i. This is a convexity argument, not a sampling assumption. Moreover,
Df(x, lambda) = Df_base(x), so the existing interval Jacobian bound on the full
box X can be reused without parameter interval evaluation.

With row bounds kappa_i for I-B*Df(X), require

    max_i kappa_i < 1,
    eta_i + kappa_i*r < r  for every i.

Together with z_j != 0 and the trace exclusion on X, these give a unique regular
root in X for every t by the uniform contraction theorem. Componentwise eta_i
is optional but sharper than the common maximum eta. Do not apply this endpoint
formula to a moving center or variable orientation without a new derivation.

## C2 deliverable and acceptance

Use root 1 and an initial piece of an archived main loop as the first prototype.
Freeze the exact dyadic parameter endpoints and reference the same exact model
as C1. Subdivision inside an archived segment is allowed; record the rational
subinterval so that its relationship to the archived path is checkable.

Deliver two consecutive nonzero segments, each with an actual uniform box
certificate. A repeated basepoint or two independently certified endpoint roots
does not satisfy this task. Fixed center tubes may need much larger radii than
the C1 boxes; those larger boxes need their own Jacobian and exclusion bounds.

For a segment ending at lambda_B compute the endpoint enclosure

    E = c - B*f(c,lambda_B) + (I-B*Df(X))*(X-c).

The right-end root belongs to E. Prove E is strictly inside the next segment's
box, and prove the original C1 root box is inside the first segment's box.
These inclusions plus uniqueness establish the root identity chain. Box
overlap or nearest-neighbor matching does not establish it. If E is too wide,
intersect it with X (both contain the root) or refine with a separately justified
endpoint enclosure, retaining the identity chain.

Negative control: replace the second tube by a valid tube for a different root
at the common parameter endpoint. Its individual contraction certificate may
pass, but the gluing acceptance must fail. Merely exchanging an unused seed
label does not test the proof checker. Also reject a parameter endpoint mismatch,
an illegal rotation target, and a tube containing z_j=0.

C2 prototyping can proceed while the C1 verifier is hardened. Final acceptance
must use the repaired complete C1 check. No full-path certification or group
upgrade follows from this two-segment prototype.
