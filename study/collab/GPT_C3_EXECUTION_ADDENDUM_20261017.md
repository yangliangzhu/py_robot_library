# C3: scale the certificate before batching all three loops

C2 implements the correct endpoint enclosure and inclusion mechanism. Its two
segments have length 2^-25 each; this establishes the proof mechanism, not its
cost over a complete loop. In particular, root 13's second tube is four times
the first radius and has kappa about 0.44. Do not extrapolate this tube growth
indefinitely or simply launch three complete loops at the prototype step size.

Continue with the same executor, in the following order.

## C3a: one complete archived segment, six roots

Use exp66 `fold3_main` (the a loop), from archived node 0 to node 1, with adaptive
rational subdivisions. Preserve the exact model and basepoint certificates.
Report accepted/rejected subdivisions, interval evaluation time, certificate
size, minimum step, and the root that limits progress. Every accepted step needs
the existing whole-tube proof; numerical prediction only chooses candidate boxes.

Avoid propagating a progressively wider endpoint enclosure. At each junction,
one may instead certify a small node box N independently at the exact endpoint,
then prove N lies strictly inside both adjacent tubes. Its unique root is a root
in each tube, so uniqueness in each tube identifies the two lifted roots. This
is a second rigorous gluing rule; it does NOT require the old endpoint enclosure
to fit inside N. Do not mistake independently certified but merely overlapping
boxes for this containment proof. Recenter each new tube near its segment's
midpoint rather than obligatorily at the previous endpoint Newton image.

## Optional exact improvement: precondition before interval evaluation

The current calculation evaluates J entries over a box before multiplying by B.
This loses algebraic cancellations. Since B is frozen rational, combine Laurent
coefficients first. For real output row i and complex variable j, define

    W_ij(z) = sum_k (B[i,2k] - i*B[i,2k+1]) * J[k,j](z).

Then the two realified entries of B*Df are

    (B*Df)[i,2j] = Re W_ij,
    (B*Df)[i,2j+1] = -Im W_ij.

Evaluate each combined Laurent polynomial by exact interval arithmetic; subtract
these entries from the real identity to bound I-B*Df. This is an exact identity,
not an approximate cancellation or a new rounding assumption. It may tighten
bounds; benchmark both methods before committing to a full-loop run. A useful
cross-check compares exact point evaluations of both formulations. Existing
certificates and the original backend remain valid.

## C3b and C3c: close one loop, then the other two

After the first full archived segment demonstrates feasible progress, certify
all six lifts around a. Then certify d and h with the same machinery. Every
subsegment must be tied to an archived edge by exact rational interpolation;
ordered subintervals must cover each edge with no gaps. Require exact loop
closure, all six base labels, and proof-based endpoint matching.

For final matching, either put the propagated endpoint enclosure inside a C1
box, or certify a small endpoint node box N contained in BOTH the last tube and
the appropriate C1 box. The latter containment again identifies the root by
uniqueness and avoids forcing a broad tube's first Newton enclosure directly
into a radius-2^-40 box. Smaller N may require more precise numerical centers;
freeze all final values exactly and recheck them.

Read permutations only from these inclusion proofs. Keep partial coverage
explicitly partial. Expected numerical permutations are a guide, not inputs to
the acceptance decision. Retain failed attempts and any refinements.

## C4 boundary

The current source review and successful recomputation are not an independently
implemented backend. Independence is a software assurance goal for C4; it is not
an extra hypothesis of the contraction theorem. A separate checker must rebuild
the exact equations and validate coverage, inclusions, exclusions, and endpoint
labels rather than trusting producer PASS fields. A second arithmetic backend
is useful but does not replace those logical checks.
