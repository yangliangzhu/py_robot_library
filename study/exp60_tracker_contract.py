#!/usr/bin/env python3
"""exp60: the tracker endpoint contract -- exact-root probe, before/after, regression suite.

GPT's math review (``study/collab/GPT_MATH_REVIEW_20261016.md`` section 3, task sheet T1) found
that ``exp38_monodromy_group.track_path`` reports success without reaching the requested
endpoint: when a step is too large the tracker shrinks the span, accepts the *intermediate*
point, and the outer loop then moves on to the next requested target; if the last segment is
shrunk as well, the function returns ``ok = True`` short of the endpoint.
``exp37_complex_loop.track_circle`` has the same outer/inner control flow and the same defect.

The probe makes that exact instead of qualitative.  ``links_at`` returns the parameter itself and
``newton_complex`` returns it as the configuration, so the tracked value *is* the requested
parameter and the residual is zero by construction: whatever the tracker returns is a parameter
value, not a configuration that might merely look close.

Every probe is run twice and printed as two columns:

* ``legacy`` -- the pre-fix control flow, kept verbatim in this module as the A/B control;
* ``module`` -- the implementation currently in the tree (imported from ``exp38``/``exp37``).

Before the fix the two columns agree and the contract checks fail; after it the module column
reaches the requested endpoint.  The circle probes additionally check the **unwrapped** walked
angle: ``exp(i theta)`` coming back to the base point is not proof that the loop was walked, so
the swept angle is reconstructed from the accepted chain and compared with
``orientation * 2 pi``.

Run::

    python3 -m study.exp60_tracker_contract

Exit code 0 iff every contract check on the current implementation passes.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest.mock import patch

import numpy as np

from study.exp37_complex_loop import links_at, newton_complex, track_circle
from study.exp38_monodromy_group import track_path

PATH_MODULE = "study.exp38_monodromy_group"
CIRCLE_MODULE = "study.exp37_complex_loop"
#: the legacy control lives in this module; ``__name__`` is ``__main__`` under ``python3 -m``
SELF_MODULE = __name__


class ExactRootCorrector:
    """Fake corrector whose exact root is the parameter value: ``q(s) = [s]``.

    ``links_at`` returns the parameter itself and ``newton_complex`` returns it as the
    configuration, so the tracker's returned value is the parameter it stopped at.  Every trial
    is recorded as ``(q_in, s_try, residual)`` so the accepted chain can be replayed.
    ``converges`` selects where the corrector is allowed to succeed, which is how the step floor
    and a mid-path breakdown are driven.
    """

    def __init__(self, converges: Callable[[complex], bool] | None = None) -> None:
        self.converges = converges if converges is not None else (lambda _s: True)
        self.calls: list[tuple[complex, complex, float]] = []

    def links_at(self, base_links, offset: float, s: complex):
        """Return the parameter value itself in place of the link transforms."""
        return complex(s)

    def newton_complex(self, links, q: np.ndarray, target, **_ignored):
        """Return the parameter as the exact root, recording the trial."""
        s = complex(links)
        residual = 0.0 if self.converges(s) else 1.0
        self.calls.append((complex(np.ravel(q)[0]), s, residual))
        return np.array([s], dtype=complex), residual


@contextmanager
def fake_corpus(module: str, fake: ExactRootCorrector):
    """Patch ``links_at`` and ``newton_complex`` in ``module`` with the exact-root fake.

    Args:
        module: Import name of the module whose globals the tracker under test reads.
        fake: The recording fake corrector.

    Yields:
        The fake, so the caller can read its call log inside the ``with`` block.
    """
    with patch(f"{module}.links_at", fake.links_at), \
            patch(f"{module}.newton_complex", fake.newton_complex):
        yield fake


def accepted_chain(calls: list[tuple[complex, complex, float]], q0: complex, *, max_step: float,
                   tolerance: float = 1e-10) -> list[complex]:
    """Replay the accepted parameter chain from the recorded corrector trials.

    The tracker only moves on when the candidate step is at most ``max_step`` and the residual
    is below ``tolerance``, and its next trial starts from that accepted configuration, so a
    forward scan rebinds the accepted sub-steps and skips the refused trials.
    """
    chain = [complex(q0)]
    for q_in, s_try, residual in calls:
        if (abs(q_in - chain[-1]) < 1e-15 and residual < tolerance
                and abs(s_try - q_in) <= max_step + 1e-15):
            chain.append(complex(s_try))
    return chain


def walked_angle(chain: list[complex], centre: complex) -> float:
    """Total signed angle swept by an accepted chain around ``centre``, unwrapped.

    Each accepted step is small (bounded by ``max_step`` against the circle radius), so the
    principal value of every consecutive angle difference is the true sweep and the sum is the
    unwrapped angle -- the quantity ``exp(i theta)`` equality cannot see.  No accepted point may
    sit exactly at ``centre`` (the probe circles have a positive radius).
    """
    total = 0.0
    for first, second in enumerate(chain[1:]):
        total += float(np.angle((second - centre) / (chain[first] - centre)))
    return total


@dataclass
class PathWalk:
    """What one ``track_path`` probe produced."""

    final: complex
    ok: bool
    chain: list[complex] = field(default_factory=list)
    calls: int = 0


@dataclass
class CircleWalk:
    """What one ``track_circle`` probe produced."""

    final: complex
    signal: float
    walked: float
    chain: list[complex] = field(default_factory=list)
    calls: int = 0


def run_path(tracker, module: str, targets: list[complex], *, q0: complex, max_step: float,
             converges: Callable[[complex], bool] | None = None) -> PathWalk:
    """Track the exact root along ``targets`` with ``tracker`` and report what came back."""
    fake = ExactRootCorrector(converges)
    with fake_corpus(module, fake):
        q, _worst, ok = tracker([], 1.0, np.eye(4), np.array([q0], dtype=complex),
                                [complex(point) for point in targets], max_step=max_step)
    chain = accepted_chain(fake.calls, q0, max_step=max_step)
    return PathWalk(final=complex(np.ravel(q)[0]), ok=bool(ok), chain=chain, calls=len(fake.calls))


def run_circle(tracker, module: str, *, centre: complex, radius: float, samples: int,
               max_step: float, orientation: float, start_angle: float = 0.0,
               converges: Callable[[complex], bool] | None = None) -> CircleWalk:
    """Walk the exact root around ``centre`` with ``tracker`` and report the unwrapped sweep."""
    start = centre + radius * np.exp(1j * start_angle)
    fake = ExactRootCorrector(converges)
    with fake_corpus(module, fake):
        q, _worst, signal = tracker([], 1.0, np.eye(4), np.array([start], dtype=complex),
                                    centre, radius, samples=samples, orientation=orientation,
                                    max_step=max_step, start_angle=start_angle)
    chain = accepted_chain(fake.calls, start, max_step=max_step)
    return CircleWalk(final=complex(np.ravel(q)[0]), signal=float(signal),
                      walked=walked_angle(chain, centre), chain=chain, calls=len(fake.calls))


def legacy_track_path(base_links, offset: float, target: np.ndarray, q0: np.ndarray,
                      s_points: list[complex], *, max_step: float = 0.1,
                      tolerance: float = 1e-10) -> tuple[np.ndarray, float, bool]:
    """The pre-fix ``exp38_monodromy_group.track_path``, kept verbatim as the A/B control.

    The inner halving loop breaks after accepting a shrunk sub-step and the outer loop then
    advances to the next requested target, so a shrunk last segment returns success short of
    the endpoint.
    """
    q = np.asarray(q0, dtype=complex)
    worst = 0.0
    s_prev = complex(s_points[0])
    for s_goal in s_points[1:]:
        span, accepted = s_goal - s_prev, False
        for _ in range(8):
            s_mid = s_prev + span
            candidate, residual = newton_complex(links_at(base_links, offset, s_mid), q, target)
            step = float(np.linalg.norm(candidate - q))
            if step <= max_step and residual < tolerance:
                q, worst, accepted = candidate, max(worst, residual), True
                s_prev = s_mid
                break
            span *= 0.5
        if not accepted:
            return q, worst, False
    return q, worst, True


def legacy_track_circle(base_links, offset: float, target: np.ndarray, q0: np.ndarray,
                        centre: float, radius: float, *, samples: int = 96,
                        orientation: float = 1.0, max_step: float = 0.1,
                        tolerance: float = 1e-10, start_angle: float = 0.0
                        ) -> tuple[np.ndarray, float, float]:
    """The pre-fix ``exp37_complex_loop.track_circle``, kept verbatim as the A/B control.

    Same defect as ``legacy_track_path``; the third return value is the *distance* to the start
    root, which is finite for a partial loop, so a caller that only tests ``isfinite`` cannot
    tell a finished loop from an abandoned one.
    """
    q = np.asarray(q0, dtype=complex)
    worst = 0.0
    angle_prev = start_angle
    for k in range(1, samples + 1):
        angle = start_angle + orientation * 2.0 * np.pi * k / samples
        span, accepted = angle - angle_prev, False
        for _ in range(8):
            mid = angle_prev + span
            s = centre + radius * np.exp(1j * mid)
            candidate, residual = newton_complex(links_at(base_links, offset, s), q, target)
            step = float(np.linalg.norm(candidate - q))
            if step <= max_step and residual < tolerance:
                q, worst, accepted = candidate, max(worst, residual), True
                angle_prev = mid
                break
            span *= 0.5
        if not accepted:
            return q, worst, float("inf")
    return q, worst, float(np.linalg.norm(q - np.asarray(q0, dtype=complex)))


def fmt(value: complex) -> str:
    """Compact formatting of a parameter value for the tables."""
    value = complex(value)
    if abs(value.imag) < 1e-12:
        return f"{value.real:.4f}"
    return f"{value.real:.4f}{value.imag:+.4f}j"


class Report:
    """Tally the contract checks and print one row per probe."""

    def __init__(self) -> None:
        self.checks = 0
        self.failures = 0

    def row(self, name: str, legacy: str, module: str, ok: bool, note: str) -> None:
        """Print one probe row and count the verdict."""
        self.checks += 1
        if not ok:
            self.failures += 1
        print(f"  {name:44s} {legacy:>26s} | {module:>26s}  "
              f"{'PASS' if ok else 'FAIL'}  {note}")


def reached(point: complex) -> Callable[[PathWalk], bool]:
    """Verdict: the tracker reported success and stopped exactly at ``point``."""
    return lambda walk: walk.ok and abs(walk.final - complex(point)) < 1e-9


def path_case(report: Report, name: str, targets: list[complex], *, q0: complex, max_step: float,
              verdict: Callable[[PathWalk], bool], converges: Callable[[complex], bool] | None
              = None, note: str = "") -> None:
    """Run one ``track_path`` probe against both implementations and check the current one."""
    legacy = run_path(legacy_track_path, SELF_MODULE, targets, q0=q0, max_step=max_step,
                      converges=converges)
    module = run_path(track_path, PATH_MODULE, targets, q0=q0, max_step=max_step,
                      converges=converges)
    report.row(name, f"{fmt(legacy.final)} ok={legacy.ok}", f"{fmt(module.final)} ok={module.ok}",
               verdict(module), note)


def circle_case(report: Report, name: str, *, centre: complex, radius: float, samples: int,
                max_step: float, orientation: float, start_angle: float = 0.0,
                verdict: Callable[[CircleWalk], bool], converges: Callable[[complex], bool] | None
                = None, note: str = "") -> None:
    """Run one ``track_circle`` probe against both implementations and check the current one."""
    legacy = run_circle(legacy_track_circle, SELF_MODULE, centre=centre, radius=radius,
                        samples=samples, max_step=max_step, orientation=orientation,
                        start_angle=start_angle, converges=converges)
    module = run_circle(track_circle, CIRCLE_MODULE, centre=centre, radius=radius,
                        samples=samples, max_step=max_step, orientation=orientation,
                        start_angle=start_angle, converges=converges)
    report.row(name, f"swept {legacy.walked:+.4f}", f"swept {module.walked:+.4f}",
               verdict(module), note)


def path_suite(report: Report) -> None:
    """The ``track_path`` contract: endpoint reached, corners kept, floor refused."""
    print("\ntrack_path -- the parameter actually reached (probe root q(s) = s)")
    print(f"  {'case':44s} {'legacy control':>26s} | {'current module':>26s}  check")
    path_case(report, "[1] exact root, targets [0, 1]", [0, 1], q0=0, max_step=0.1,
              verdict=reached(1.0), note="GPT's reproduction")
    corner = [0, 0.5, 0.5 + 0.5j, 1.0 + 0.5j]

    def corners_kept(walk: PathWalk) -> bool:
        gaps = [min(abs(point - seen) for seen in walk.chain) for point in corner]
        return walk.ok and abs(walk.final - corner[-1]) < 1e-9 and max(gaps) < 1e-12

    path_case(report, "[2] cornered polyline (no chord)", list(corner), q0=0, max_step=0.1,
              verdict=corners_kept, note="every corner visited exactly")
    reverse = [1.0 + 0.5j, 0.5 + 0.5j, 0.5, 0]

    def reversed_done(walk: PathWalk) -> bool:
        gaps = [min(abs(point - seen) for seen in walk.chain) for point in reverse]
        return walk.ok and abs(walk.final - 0) < 1e-9 and max(gaps) < 1e-12

    path_case(report, "[3] the same path walked backwards", list(reverse), q0=reverse[0],
              max_step=0.1, verdict=reversed_done, note="reverse order of [2]")
    path_case(report, "[4] last segment needs subdivision", [0, 0.05, 1.0], q0=0, max_step=0.1,
              verdict=reached(1.0), note="legacy stops at 0.109375")

    def floored(walk: PathWalk) -> bool:
        return (not walk.ok) and walk.final.real < 0.5 and 0.5 - walk.final.real < 1e-6

    path_case(report, "[5] corrector dies past s = 0.5", [0, 1], q0=0, max_step=0.1,
              verdict=floored, converges=lambda s: s.real < 0.5,
              note="ok=False at the last parameter reached")

    def refused_start(walk: PathWalk) -> bool:
        return (not walk.ok) and abs(walk.final) < 1e-12

    path_case(report, "[6] corrector never converges", [0, 1], q0=0, max_step=0.1,
              verdict=refused_start, converges=lambda _s: False, note="no progress, ok=False")

    def single_point(walk: PathWalk) -> bool:
        return walk.ok and abs(walk.final - 0.7) < 1e-12

    path_case(report, "[7] single-point path is a no-op", [0.7], q0=0.7, max_step=0.1,
              verdict=single_point, note="q0 returned unchanged")


def circle_suite(report: Report) -> None:
    """The ``track_circle`` contract: the requested arc is walked, in the requested direction."""
    print("\ntrack_circle -- the unwrapped angle actually swept (probe root q(s) = s)")
    print(f"  {'case':44s} {'legacy control':>26s} | {'current module':>26s}  check")
    radius, max_step = 1e-3, 1e-5

    def full_turn(orientation: float) -> Callable[[CircleWalk], bool]:
        def verdict(walk: CircleWalk) -> bool:
            return (abs(walk.walked - orientation * 2.0 * np.pi) < 1e-9
                    and walk.signal == 0.0)
        return verdict

    circle_case(report, "[8] full turn, orientation +1, 96 samples", centre=0, radius=radius,
                samples=96, max_step=max_step, orientation=1.0, verdict=full_turn(1.0),
                note="swept angle must be +2pi")
    circle_case(report, "[9] full turn, orientation -1, 96 samples", centre=0, radius=radius,
                samples=96, max_step=max_step, orientation=-1.0, verdict=full_turn(-1.0),
                note="endpoint equals the base point; sweep is -2pi")
    circle_case(report, "[10] full turn from only 8 samples", centre=0, radius=radius,
                samples=8, max_step=max_step, orientation=1.0, verdict=full_turn(1.0),
                note="inner sub-stepping must still finish")

    def half_only(walk: CircleWalk) -> bool:
        return walk.signal > 0.0 and walk.walked < 2.0 * np.pi - 1e-6

    circle_case(report, "[11] corrector dies in the lower half-plane", centre=0, radius=radius,
                samples=96, max_step=max_step, orientation=1.0,
                converges=lambda s: s.imag >= -1e-15, verdict=half_only,
                note="partial loop must report the outstanding angle")


def main() -> int:
    """Run the probe suite and print the before/after columns."""
    print("exp60: tracker endpoint contract -- exact-root probe q(s) = s and regression suite")
    print(f"  current implementation: {track_path.__module__}.track_path, "
          f"{track_circle.__module__}.track_circle")
    print("  legacy control: the pre-fix control flow, kept verbatim in this module")
    report = Report()
    path_suite(report)
    circle_suite(report)
    print(f"\n{report.checks - report.failures}/{report.checks} contract checks pass")
    if report.failures:
        print("FAIL: the tracker still returns early, or refuses a path it should walk")
        return 1
    print("PASS: every requested endpoint was reached, or refused with the parameter last reached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
