#!/usr/bin/env python3
"""exp01: what does ``IkStandard``'s ``ok=False`` actually mean?

The census work needs to know how a solver fails, because "no solution found" and "stopped
0.01 degrees short" are very different things for a branch study -- and for a control loop.
This experiment replicates the solver's loop with two extra observations -- the best iterate it
ever visited, and how the step size evolved -- and checks the replication against the library
class before reporting anything from it.

Measured:

1. the pose error of the *returned* configuration on failures, versus the best iterate the
   algorithm had already reached;
2. the singular value at those configurations, i.e. whether the failures cluster near the
   singular set;
3. the step-size trajectory, which is monotone non-increasing by construction.

Run::

    python3 -m study.exp01_ik_standard_failure --robot sr5 --samples 300
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType, ModelFactory
from model.ik_solver import IkStandard, _damped_pinv
from study.ik_structure import pose_error, random_reachable_target, sigma_min


class TracingIkStandard(IkStandard):
    """``IkStandard`` with the two things it does not report: best iterate, step history.

    The loop is copied from the class on purpose.  A different algorithm would measure a
    different thing, so the copy is verified against the original on the same seeds before any
    number from it is used.
    """

    def __call__(self, angle: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        """Run the solver, recording the best iterate and the step-size history."""
        tar_t = target
        max_iter = 20.0
        max_iter_sup = 50
        pre_t = self.model.fk(angle)
        t = 0
        diff = self.diff_func(pre_t, tar_t)
        dis1 = np.linalg.norm(diff[:3])
        dis2 = np.linalg.norm(diff[3:])
        eps = 1e-5
        err_norm = max(dis1, dis2)
        self.steps = []
        self.best_error = err_norm
        if err_norm < eps:
            return angle, True

        last_err_norm = 1e5
        step_size = 1.0
        pos_step_ = 1e-1
        rot_step_ = 5e-2
        q_best = np.array(angle)
        while t < max_iter:
            if err_norm > last_err_norm:
                factor = min(0.2, last_err_norm / err_norm)
                max_iter = t + 1 + (max_iter - t - 1) / factor
                max_iter = min(max_iter_sup, max_iter)
                step_size *= factor
                self.steps.append(step_size)
                if step_size < 1e-4:
                    self.returned_error = err_norm
                    self.best_q = q_best
                    return angle, False
                angle = np.array(q_best)
            else:
                q_best = np.array(angle)
                jac = self.jac_func(angle)
                inv_jac = _damped_pinv(jac, singular_threshold=0.01)
                if dis1 > pos_step_:
                    diff[:3] = diff[:3] * pos_step_ / dis1
                if dis2 > rot_step_:
                    diff[3:] = diff[3:] * rot_step_ / dis2
                dq = inv_jac @ diff.flatten()
            angle = angle + dq * step_size
            angle = np.clip(angle, self.model.lower_bounds, self.model.upper_bounds)
            pre_t = self.model.fk(angle)
            diff = self.diff_func(pre_t, tar_t)
            dis1 = np.linalg.norm(diff[:3])
            dis2 = np.linalg.norm(diff[3:])
            last_err_norm = err_norm
            err_norm = max(dis1, dis2)
            self.best_error = min(self.best_error, err_norm)
            if err_norm < eps:
                return angle, True
            t = t + 1
        self.returned_error = err_norm
        self.best_q = q_best
        return angle, False


def main() -> int:
    """Run the failure-path measurements."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="sr5")
    parser.add_argument("--backend", default="casadi")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--near-singular", type=float, default=0.02,
                        help="targets whose solution has sigma_min below this are the suspect set")
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend=args.backend, ik_type=IkType.IK_STANDARD)
    tracing = TracingIkStandard(model, IkType.IK_STANDARD)
    library = IkStandard(model, IkType.IK_STANDARD)
    rng = np.random.default_rng(args.seed)

    # The replication must agree with the class, or the numbers below measure the copy.
    target_check, known_check = random_reachable_target(model, rng)
    mismatches = 0
    for _ in range(20):
        seed = known_check + rng.normal(0.0, 0.2, model.num_dof)
        q_lib, ok_lib = library(seed, target_check)
        q_trace, ok_trace = tracing(seed, target_check)
        mismatches += int(ok_lib != ok_trace or not np.allclose(q_lib, q_trace, atol=1e-12))
    print(f"replication check: {20 - mismatches}/20 seeds agree with IkStandard "
          f"({'ok' if mismatches == 0 else 'MISMATCH -- numbers below are suspect'})")

    failures, suspect, gaps, bests, sigmas, step_ends = 0, 0, [], [], [], []
    for _ in range(args.samples):
        target, known = random_reachable_target(model, rng)
        seed = known + rng.normal(0.0, 0.2, model.num_dof)
        q, ok = tracing(seed, target)
        if ok:
            continue
        failures += 1
        returned = max(pose_error(model, q, target))
        best = tracing.best_error
        gaps.append(returned - best)
        bests.append(best)
        sigmas.append(sigma_min(model, tracing.best_q))
        step_ends.append(tracing.steps[-1] if tracing.steps else float("nan"))
        if best < 1e-4:
            suspect += 1
    if failures == 0:
        print("no failures in the sample; nothing to report")
        return 0
    gaps = np.asarray(gaps)
    bests = np.asarray(bests)
    sigmas = np.asarray(sigmas)
    print(f"\n{failures} failures out of {args.samples} seeds (0.2 rad perturbations):")
    print(f"  best pose error the solver had reached: median {np.median(bests):.2e}, "
          f"90th pct {np.percentile(bests, 90):.2e}, max {np.max(bests):.2e}")
    print(f"  returned minus best: median {np.median(gaps):+.2e}, "
          f"worst overshoot {np.max(gaps):.2e}, worst undershoot {np.min(gaps):+.2e}")
    print(f"  failures that were already within 1e-4 of the target: {suspect}/{failures} "
          f"({suspect / failures:.0%})")
    print(f"  sigma_min of the best iterate: median {np.median(sigmas):.3f}, "
          f"below {args.near_singular}: {(sigmas < args.near_singular).sum()}/{failures}")
    print(f"  final step size before giving up: median {np.nanmedian(step_ends):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
