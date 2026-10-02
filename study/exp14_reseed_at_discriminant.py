#!/usr/bin/env python3
"""exp14: closing the homotopy's gap -- reseeding at the discriminant

exp13 measured the limit of a single-parameter-path tracker: where the path meets the discriminant
a branch dies, and the solutions that exist at the real arm but not on any branch connected to the
seeds are simply not reachable by following branches.  Note what makes that gap closable at all:
**the target pose never moves**, only the DH parameters do.  At every ``s`` the current arm has a
whole fibre of solutions for the same pose, so when a tracked branch dies there are other branches
present at that same ``s`` -- they were born at an earlier discriminant, and a multi-start solve of
the *intermediate* arm finds them.

So this experiment does exactly that: track from SR0's closed-form fibre; wherever a track stalls,
solve the intermediate arm from scratch (a local census), and continue every solution that is not
already being tracked.  It measures how much of the real arm's fibre that recovers, against an
800-seed census of the real arm as the reference.  The recipe is hybrid on purpose, and its
coverage is the number that says whether the hybrid is enough or whether branch switching proper
(deflation, or a complex path) is needed.

Run::

    python3 -m study.exp14_reseed_at_discriminant --poses 2 --seeds 800
"""

from __future__ import annotations

import argparse

import numpy as np

from study import homotopy, sr0
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at


def path_from(offset: float, s0: float):
    """The same path, reparameterised so its own ``s = 0`` is the original ``s0``."""
    straight = path_at(offset)

    def make(s: float) -> list[np.ndarray]:
        return straight(s0 + s * (1.0 - s0))

    return make


def main() -> int:
    """Run the reseeding experiment."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=800)
    parser.add_argument("--local-seeds", type=int, default=600,
                        help="multi-start seeds for the intermediate-arm census")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match", type=float, default=1e-4)
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr0_model = sr0.robot()
    sr5_model = sr0.robot(link_transforms())  # the real arm
    circular = iks.circular_joints(sr5_model)
    rng = np.random.default_rng(args.seed)
    print(f"SR0 -> SR5 along link 5 y: 0 -> {offset:.4f} m, reseeding at every stalled track")

    for pose in range(args.poses):
        q_true = rng.uniform(-2.0, 2.0, sr5_model.num_dof)
        target = sr5_model.fk(q_true)
        seeds = sr0.solve(sr0_model, target)

        # stage 1: the straight path, exactly as in exp13
        tracks = [
            homotopy.track(path_at(offset), target, seed.q, sigma_floor=args.sigma_floor)
            for seed in seeds
        ]
        stalled = [t for t in tracks if not t.finished]

        # stage 2: at each stall, solve the intermediate arm from scratch and continue anything new
        reseeded = 0
        for track in stalled:
            s_star = track.progress
            intermediate = sr0.robot(path_at(offset)(s_star))
            fiber = census(
                intermediate, target, solver=make_lm_solver(intermediate),
                seeds=args.local_seeds, rng=rng,
            )
            for candidate in fiber.solutions:
                if any(
                    iks.configuration_distance(candidate, other.q, circular) < args.match
                    for other in tracks
                ):
                    continue
                attempt = homotopy.track(
                    path_from(offset, s_star), target, candidate, sigma_floor=args.sigma_floor
                )
                attempt.reason = f"reseeded at s = {s_star:.3f}: {attempt.reason}"
                tracks.append(attempt)
                reseeded += 1

        arrived = [t for t in tracks if t.finished]
        fiber = census(sr5_model, target, solver=make_lm_solver(sr5_model), seeds=args.seeds, rng=rng)
        numerical = fiber.solutions

        def matches(point: np.ndarray, others: list[np.ndarray]) -> bool:
            return any(
                iks.configuration_distance(point, other, circular) < args.match for other in others
            )

        reached = [t.q for t in arrived]
        missed = [q for q in numerical if not matches(q, reached)]
        extra = [q for q in reached if not matches(q, numerical)]
        print(
            f"pose {pose}: seeds {len(seeds)} | straight path arrived "
            f"{len([t for t in tracks[: len(seeds)] if t.finished])} | reseeded tracks {reseeded} "
            f"| total arrived {len(arrived)} | census {len(numerical)} "
            f"| census solutions still missed: {len(missed)} "
            f"| homotopy-only solutions: {len(extra)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
