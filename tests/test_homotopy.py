"""Tests for src/swarm/homotopy.py.

Includes a finite-difference oracle for homotopy_penalty_and_grad — the
SAME pattern used in test_minco_gradients_e2e.py to keep gradient bugs
from hiding."""

from __future__ import annotations

import numpy as np
import pytest

from src.swarm.homotopy import (
    HomotopyPenaltyContext,
    build_penalty_context,
    full_signature,
    generate_target_signatures,
    homotopy_penalty_and_grad,
    pairwise_signature,
)


# ---------------------------------------------------------------------------
# Signature correctness
# ---------------------------------------------------------------------------

class TestPairwiseSignature:

    def test_no_interaction_returns_zero(self) -> None:
        """Two drones far apart should return signature 0."""
        own = np.array([[0., 0., 1.], [10., 0., 1.]])
        own_t = np.array([0., 1.])
        nbr = np.array([[0., 100., 1.], [10., 100., 1.]])
        nbr_t = np.array([0., 1.])
        sig = pairwise_signature(own, own_t, nbr, nbr_t, interaction_radius=3.0)
        assert sig == 0

    def test_pass_on_left_returns_plus_one(self) -> None:
        """Own travels parallel on the +y side of neighbour going +x → +1.

        Using a non-crossing offset trajectory (not an exact crossing) so
        the closest-approach geometry isn't degenerate: rel at closest
        approach is (0, +offset), not (0, 0)."""
        nbr = np.array([[0., 0., 1.], [10., 0., 1.]])
        nbr_t = np.array([0., 1.])
        own = np.array([[0., 1.5, 1.], [10., 1.5, 1.]])
        own_t = np.array([0., 1.])
        sig = pairwise_signature(own, own_t, nbr, nbr_t, interaction_radius=3.0)
        assert sig == +1

    def test_pass_on_right_returns_minus_one(self) -> None:
        nbr = np.array([[0., 0., 1.], [10., 0., 1.]])
        nbr_t = np.array([0., 1.])
        own = np.array([[0., -1.5, 1.], [10., -1.5, 1.]])
        own_t = np.array([0., 1.])
        sig = pairwise_signature(own, own_t, nbr, nbr_t, interaction_radius=3.0)
        assert sig == -1

    def test_winding_full_circle_ccw_returns_plus_one(self) -> None:
        """Own loops around stationary neighbour CCW → +1."""
        nbr = np.array([[0., 0., 1.]] * 32)
        nbr_t = np.linspace(0., 1., 32)
        # CCW circle of radius 2
        angles = np.linspace(0., 2*np.pi, 32)
        own = np.stack([2*np.cos(angles), 2*np.sin(angles), np.ones_like(angles)], axis=1)
        own_t = np.linspace(0., 1., 32)
        sig = pairwise_signature(own, own_t, nbr, nbr_t, interaction_radius=10.0)
        assert sig == +1

    def test_winding_full_circle_cw_returns_minus_one(self) -> None:
        nbr = np.array([[0., 0., 1.]] * 32)
        nbr_t = np.linspace(0., 1., 32)
        angles = np.linspace(0., -2*np.pi, 32)
        own = np.stack([2*np.cos(angles), 2*np.sin(angles), np.ones_like(angles)], axis=1)
        own_t = np.linspace(0., 1., 32)
        sig = pairwise_signature(own, own_t, nbr, nbr_t, interaction_radius=10.0)
        assert sig == -1


class TestFullSignature:

    def test_signature_per_neighbour_independent(self) -> None:
        """Own on +y side of nbr1, on -y side of nbr2 → opposite signs.

        Both neighbours go +x. Own's y position is between them. The
        lateral CCW direction for both is +y, so the sign of (own_y -
        nbr_y) determines the signature directly."""
        own = np.array([[0., 0., 1.], [10., 0., 1.]])    # own at y=0
        own_t = np.array([0., 1.])
        nbr1 = np.array([[0., -1.5, 1.], [10., -1.5, 1.]])  # below own → own is +y of nbr1
        nbr1_t = np.array([0., 1.])
        nbr2 = np.array([[0., +1.5, 1.], [10., +1.5, 1.]])  # above own → own is -y of nbr2
        nbr2_t = np.array([0., 1.])
        sig = full_signature(own, own_t, [(nbr1, nbr1_t), (nbr2, nbr2_t)],
                              interaction_radius=3.0)
        assert len(sig) == 2
        assert sig[0] == +1 and sig[1] == -1

    def test_empty_neighbours_returns_empty_tuple(self) -> None:
        own = np.array([[0., 0., 1.], [10., 0., 1.]])
        sig = full_signature(own, np.array([0., 1.]), [])
        assert sig == ()


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------

class TestGenerateTargetSignatures:

    def test_single_neighbour_one_flip(self) -> None:
        """One interactive neighbour, 4 branches requested → 1 flip available."""
        out = generate_target_signatures((+1,), n_branches=4)
        assert out == [(-1,)]

    def test_two_neighbours_three_flips(self) -> None:
        """Two interactive neighbours: single-flips first, then double-flip."""
        out = generate_target_signatures((+1, +1), n_branches=4)
        # Single flips: (-,+), (+,-), then double flip (-,-)
        assert (-1, +1) in out
        assert (+1, -1) in out
        assert (-1, -1) in out
        assert len(out) == 3

    def test_zero_signs_not_flipped(self) -> None:
        """Non-interacting neighbours (sign 0) shouldn't be flipped."""
        out = generate_target_signatures((+1, 0, -1), n_branches=10)
        # Only positions 0 and 2 are flippable
        for sig in out:
            assert sig[1] == 0  # middle stays 0

    def test_caps_at_n_branches(self) -> None:
        out = generate_target_signatures((+1, +1, +1, +1), n_branches=3)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# Penalty gradient finite-difference oracle
# ---------------------------------------------------------------------------

class TestPenaltyGradient:

    @pytest.fixture
    def simple_ctx(self):
        """Single neighbour at (5, 0, 1) going +x, own waypoint near it."""
        # Two interior waypoints at times t=1.0 and t=2.0
        interior_times = np.array([1.0, 2.0])
        # Neighbour going +x: at t=1, neighbour is at (3, 0, 1); at t=2, (6, 0, 1)
        nxyz = np.array([[0., 0., 1.], [3., 0., 1.], [6., 0., 1.], [10., 0., 1.]])
        nts = np.array([0., 1., 2., 3.])
        neighbours = [(nxyz, nts)]
        # Target sign +1: own should be on +y side
        ctx = build_penalty_context(
            interior_waypoint_times=interior_times,
            neighbours=neighbours,
            target_signature=(+1,),
            weight=1.0e3,
            epsilon=0.1,
        )
        return ctx

    def test_no_violation_on_correct_side_returns_zero(self, simple_ctx) -> None:
        """Waypoints on +y side of neighbour → zero penalty."""
        wps = np.array([[3., 5., 1.], [6., 5., 1.]])  # well on +y side
        cost, grad = homotopy_penalty_and_grad(wps, simple_ctx)
        assert cost == 0.0
        np.testing.assert_array_equal(grad, np.zeros_like(wps))

    def test_violation_on_wrong_side_gives_positive_cost(self, simple_ctx) -> None:
        """Waypoints on -y side → positive cost, gradient pulls them to +y."""
        wps = np.array([[3., -2., 1.], [6., -2., 1.]])
        cost, grad = homotopy_penalty_and_grad(wps, simple_ctx)
        assert cost > 0
        # Gradient on y should be NEGATIVE (so moving y up reduces cost)
        # Actually: cost = w * (-s_j * d + eps)^2 with s_j=+1, d=relative_y
        # d violation/d (own_y) for s_j=+1: when violation > 0,
        #   d cost/d(own_y) = 2 w * v * (-s_j) * lat_y = -2w v * lat_y
        # lat_y = +1 for nbr going +x (lateral CCW is +y in xy)
        # So gradient on y is -2 w v < 0
        # MOVING in -gradient direction (i.e., +y) reduces cost ✓
        assert np.all(grad[:, 1] < 0)
        # x and z components should be 0 (only y matters here)
        np.testing.assert_array_equal(grad[:, 0], np.zeros(2))
        np.testing.assert_array_equal(grad[:, 2], np.zeros(2))

    def test_gradient_matches_finite_difference(self, simple_ctx) -> None:
        """Analytical gradient vs central-difference numerical gradient."""
        rng = np.random.default_rng(42)
        wps = np.array([[3., 1.5, 1.], [6., -0.5, 1.]])  # mixed sides
        cost, grad = homotopy_penalty_and_grad(wps, simple_ctx)
        # Skip if no violation anywhere (gradient trivially 0)
        if cost == 0:
            pytest.skip("no violation; FD test uninformative")
        h = 1e-6
        grad_fd = np.zeros_like(wps)
        for i in range(wps.shape[0]):
            for d in range(wps.shape[1]):
                wp = wps.copy(); wp[i, d] += h
                c_plus, _ = homotopy_penalty_and_grad(wp, simple_ctx)
                wp = wps.copy(); wp[i, d] -= h
                c_minus, _ = homotopy_penalty_and_grad(wp, simple_ctx)
                grad_fd[i, d] = (c_plus - c_minus) / (2 * h)
        # FD oracle: match analytical to 1e-3 rtol on non-zero entries
        # (z dimension is always zero by construction)
        mask = np.abs(grad_fd) > 1e-6
        np.testing.assert_allclose(
            grad[mask], grad_fd[mask], rtol=1e-3, atol=1e-3,
            err_msg=f"analytical:\n{grad}\nFD:\n{grad_fd}",
        )

    def test_zero_signature_no_contribution(self) -> None:
        """A neighbour with target sign 0 contributes nothing to penalty."""
        interior_times = np.array([1.0, 2.0])
        nxyz = np.array([[3., 0., 1.], [6., 0., 1.]])
        nts = np.array([1., 2.])
        ctx = build_penalty_context(
            interior_waypoint_times=interior_times,
            neighbours=[(nxyz, nts)],
            target_signature=(0,),  # no constraint
            weight=1.0e3,
        )
        wps = np.array([[3., -100., 1.], [6., -100., 1.]])  # absurdly off
        cost, grad = homotopy_penalty_and_grad(wps, ctx)
        assert cost == 0.0
        np.testing.assert_array_equal(grad, np.zeros_like(wps))


# ---------------------------------------------------------------------------
# 3-D signature (Gap 1)
# ---------------------------------------------------------------------------

from src.swarm.homotopy import (  # noqa: E402
    full_signature_3d,
    generate_target_signatures_3d,
    pairwise_signature_3d,
)


class TestPairwiseSignature3D:

    def test_pass_over_returns_plus_z(self) -> None:
        """Own travels above neighbour → vertical sign +1."""
        nbr = np.array([[0., 0., 1.], [10., 0., 1.]])
        nbr_t = np.array([0., 1.])
        own = np.array([[0., 1.5, 3.0], [10., 1.5, 3.0]])  # +y, +z high
        own_t = np.array([0., 1.])
        h, v = pairwise_signature_3d(own, own_t, nbr, nbr_t, interaction_radius=3.0)
        assert h == +1
        assert v == +1

    def test_pass_under_returns_minus_z(self) -> None:
        nbr = np.array([[0., 0., 5.0], [10., 0., 5.0]])
        nbr_t = np.array([0., 1.])
        own = np.array([[0., 1.5, 1.0], [10., 1.5, 1.0]])  # +y, way below
        own_t = np.array([0., 1.])
        h, v = pairwise_signature_3d(own, own_t, nbr, nbr_t, interaction_radius=3.0)
        assert h == +1
        assert v == -1

    def test_shared_altitude_returns_zero_v(self) -> None:
        nbr = np.array([[0., 0., 1.], [10., 0., 1.]])
        nbr_t = np.array([0., 1.])
        own = np.array([[0., 1.5, 1.1], [10., 1.5, 1.1]])  # |Δz|=0.1 < band 0.5
        own_t = np.array([0., 1.])
        h, v = pairwise_signature_3d(own, own_t, nbr, nbr_t, interaction_radius=3.0)
        assert h == +1
        assert v == 0

    def test_no_interaction_returns_zero_pair(self) -> None:
        nbr = np.array([[0., 100., 1.], [10., 100., 1.]])
        nbr_t = np.array([0., 1.])
        own = np.array([[0., 0., 5.], [10., 0., 5.]])
        own_t = np.array([0., 1.])
        sig = pairwise_signature_3d(own, own_t, nbr, nbr_t, interaction_radius=3.0)
        assert sig == (0, 0)

    def test_2d_wrapper_matches_3d_horizontal(self) -> None:
        """Legacy pairwise_signature returns only the horizontal entry."""
        nbr = np.array([[0., 0., 1.], [10., 0., 1.]])
        nbr_t = np.array([0., 1.])
        own = np.array([[0., 1.5, 3.0], [10., 1.5, 3.0]])
        own_t = np.array([0., 1.])
        legacy = pairwise_signature(own, own_t, nbr, nbr_t, interaction_radius=3.0)
        h, _ = pairwise_signature_3d(own, own_t, nbr, nbr_t, interaction_radius=3.0)
        assert legacy == h


class TestGenerate3DTargetSignatures:

    def test_single_neighbour_three_flips_when_v_nonzero(self) -> None:
        # current=((+1, +1)) — flippable: flip h, flip v, flip both.
        targets = generate_target_signatures_3d(((+1, +1),), n_branches=5)
        assert ((-1, +1),) in targets
        assert ((+1, -1),) in targets
        assert ((-1, -1),) in targets
        assert ((+1, +1),) not in targets  # never the input

    def test_zero_h_neighbour_skipped(self) -> None:
        targets = generate_target_signatures_3d(((0, 0),), n_branches=4)
        assert targets == []

    def test_v_zero_emits_introduction_variants(self) -> None:
        """When v=0 (shared altitude band), introduce ±1 vertical options."""
        targets = generate_target_signatures_3d(((+1, 0),), n_branches=10)
        # flip h only, plus four (h × v) combinations with v = ±1
        assert ((-1, 0),) in targets
        assert ((-1, +1),) in targets
        assert ((-1, -1),) in targets
        assert ((+1, +1),) in targets
        assert ((+1, -1),) in targets

    def test_n_branches_cap(self) -> None:
        targets = generate_target_signatures_3d(((+1, +1), (+1, +1)), n_branches=2)
        assert len(targets) == 2


class TestVerticalPenaltyGradient:

    def test_pass_over_target_no_violation(self) -> None:
        """Target v=+1 (pass over): if waypoint z is above neighbour, no cost."""
        interior_times = np.array([1.0])
        nxyz = np.array([[3., 0., 1.0], [5., 0., 1.0]])
        nts = np.array([0.5, 1.5])
        ctx = build_penalty_context(
            interior_waypoint_times=interior_times,
            neighbours=[(nxyz, nts)],
            target_signature=((+1, +1),),
            weight=1.0e3,
            weight_v=1.0e3,
            epsilon_v=0.3,
        )
        wps = np.array([[4., 2.0, 3.0]])  # high above
        cost, grad = homotopy_penalty_and_grad(wps, ctx)
        assert cost == 0.0
        np.testing.assert_allclose(grad, np.zeros_like(wps), atol=1e-12)

    def test_pass_over_target_with_under_position_pushes_up(self) -> None:
        """Target v=+1 but waypoint is below neighbour → positive cost, grad pushes up."""
        interior_times = np.array([1.0])
        nxyz = np.array([[3., 0., 1.0], [5., 0., 1.0]])
        nts = np.array([0.5, 1.5])
        ctx = build_penalty_context(
            interior_waypoint_times=interior_times,
            neighbours=[(nxyz, nts)],
            target_signature=((+1, +1),),
            weight=0.0,         # horizontal disabled
            weight_v=1.0e3,
            epsilon_v=0.3,
        )
        wps = np.array([[4., 2.0, -2.0]])  # well below
        cost, grad = homotopy_penalty_and_grad(wps, ctx)
        assert cost > 0.0
        # rel_z = own_z - nbr_z = -3. s_j_v=+1, violation = max(0, 3 + 0.3) = 3.3.
        # d cost / d q_z = 2 w v (-s_j_v) = -negative → moving +z decreases cost.
        assert grad[0, 2] < 0.0

    def test_vertical_gradient_matches_fd(self) -> None:
        interior_times = np.array([1.0, 2.0])
        nxyz = np.array([[3., 0., 1.0], [6., 0., 1.0]])
        nts = np.array([1., 2.])
        ctx = build_penalty_context(
            interior_waypoint_times=interior_times,
            neighbours=[(nxyz, nts)],
            target_signature=((+1, -1),),  # pass-under
            weight=0.0,
            weight_v=1.0e3,
            epsilon_v=0.2,
        )
        wps = np.array([[3., 0., 2.0], [6., 0., 2.0]])  # above → violates pass-under
        cost, grad = homotopy_penalty_and_grad(wps, ctx)
        if cost == 0:
            pytest.skip("no violation; FD test uninformative")
        h = 1e-6
        grad_fd = np.zeros_like(wps)
        for i in range(wps.shape[0]):
            for d in range(wps.shape[1]):
                wp = wps.copy(); wp[i, d] += h
                c_plus, _ = homotopy_penalty_and_grad(wp, ctx)
                wp = wps.copy(); wp[i, d] -= h
                c_minus, _ = homotopy_penalty_and_grad(wp, ctx)
                grad_fd[i, d] = (c_plus - c_minus) / (2 * h)
        mask = np.abs(grad_fd) > 1e-6
        np.testing.assert_allclose(grad[mask], grad_fd[mask], rtol=1e-3, atol=1e-3)
