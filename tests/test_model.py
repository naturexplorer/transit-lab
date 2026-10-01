"""Validation tests for the transit model (stage 3)."""

import math
from typing import Any
import batman
import numpy as np
import pytest

from transit_lab.model import flux_quadratic, flux_uniform, kappa, separation


def batman_flux(t, p, a_over_rstar, inc_deg, g1=None, g2=None, period=3.0, t0=0.0):
    """Reference light curve from batman for a circular orbit.

    Quadratic limb darkening if g1, g2 are given, otherwise a uniform source.
    """
    params: Any = batman.TransitParams()
    params.t0 = t0                # time of mid-transit
    params.per = period           # orbital period
    params.rp = p                 # planet radius / stellar radius
    params.a = a_over_rstar       # orbital radius / stellar radius
    params.inc = inc_deg          # inclination in DEGREES
    params.ecc = 0.0              # circular orbit
    params.w = 90.0               # irrelevant for a circular orbit
    if g1 is None:
        params.limb_dark = "uniform"
        params.u = []
    else:
        params.limb_dark = "quadratic"
        params.u = [g1, g2]       # batman's u1, u2 are our gamma_1, gamma_2
    return batman.TransitModel(params, t).light_curve(params)


@pytest.mark.parametrize("p", [0.01, 0.1, 0.3])
def test_uniform_depth_matches_radius_ratio(p):
    """Mid-transit depth of a central transit equals p**2."""
    assert math.isclose(1 - flux_uniform(p, 0.0), p**2)


@pytest.mark.parametrize(
    "p, z, expected",
    [
        (0.1, 1.2, 1.0),   # no overlap
        (0.1, 0.5, 0.99),  # planet fully inside the star: 1 - p^2
        (1.5, 0.3, 0.0),   # star fully covered
    ],
)
def test_uniform_simple_cases(p, z, expected):
    """The three closed-form cases of the uniform source."""
    assert math.isclose(flux_uniform(p, z), expected)


@pytest.mark.parametrize("p", [0.1, 0.5, 1.5])
def test_uniform_continuous_at_contacts(p):
    """The partial-overlap formula joins the simple cases at z = |1-p| and z = 1+p."""
    eps = 1e-9
    for z0 in [abs(1 - p), 1 + p]:
        assert math.isclose(flux_uniform(p, z0 - eps), flux_uniform(p, z0 + eps), abs_tol=1e-6)


@pytest.mark.parametrize("p", [0.3, 0.8, 1.5])
@pytest.mark.parametrize("frac", [0.1, 0.5, 0.9])
def test_uniform_swap_symmetry(p, frac):
    """Swapping star and planet leaves the overlap area unchanged.
    Area = pi * lam(p, z) = p^2 * pi * lam(1/p, z/p), with lam = 1 - flux_uniform.
    """
    z = abs(1 - p) + frac * (1 + p - abs(1 - p))   # inside the partial-overlap range
    lam = 1 - flux_uniform(p, z)
    lam_swapped = 1 - flux_uniform(1 / p, z / p)
    assert math.isclose(lam, p**2 * lam_swapped, rel_tol=1e-9)


@pytest.mark.parametrize(
    "p, a_over_rstar, inc_deg",
    [(0.1, 15.0, 90.0), (0.1, 15.0, 87.0), (0.1, 10.0, 84.3), (0.2, 8.0, 89.0)],
)
def test_uniform_matches_batman(p, a_over_rstar, inc_deg):
    """Closed-form uniform source agrees with batman's uniform model."""
    period, t0 = 3.0, 0.0
    t = np.linspace(-0.1, 0.1, 500)
    inc_rad = math.radians(inc_deg)
    ours = [flux_uniform(p, separation(ti, period, t0, a_over_rstar, inc_rad)) for ti in t]
    ref = batman_flux(t, p, a_over_rstar, inc_deg, period=period, t0=t0)
    assert np.max(np.abs(np.array(ours) - ref)) < 1e-9


@pytest.mark.parametrize("p", [0.1, 0.5])
def test_quadratic_reduces_to_uniform(p):
    """With no limb darkening, the ring integral reproduces the closed-form uniform source."""
    for z in np.linspace(0.0, 1 + p + 0.1, 25):
        assert math.isclose(flux_quadratic(p, z, 0.0, 0.0), flux_uniform(p, z), abs_tol=1e-6)


@pytest.mark.parametrize(
    "p, a_over_rstar, inc_deg, g1, g2",
    [
        (0.1, 15.0, 90.0, 0.40, 0.25),   # central transit
        (0.1, 15.0, 87.0, 0.40, 0.25),   # off-centre (b ~ 0.79)
        (0.1, 10.0, 84.3, 0.40, 0.25),   # grazing (b ~ 0.99)
        (0.2, 8.0, 89.0, 0.60, 0.10),    # large planet, strong darkening
        (0.05, 20.0, 88.0, 0.30, 0.30),  # small planet
    ],
)
def test_quadratic_matches_batman(p, a_over_rstar, inc_deg, g1, g2):
    """Agreement with batman to ~1e-6 across a parameter sweep."""
    period, t0 = 3.0, 0.0
    t = np.linspace(-0.1, 0.1, 500)
    inc_rad = math.radians(inc_deg)
    ours = [
        flux_quadratic(p, separation(ti, period, t0, a_over_rstar, inc_rad), g1, g2)
        for ti in t
    ]
    ref = batman_flux(t, p, a_over_rstar, inc_deg, g1, g2, period, t0)
    assert np.max(np.abs(np.array(ours) - ref)) < 1e-6


def floats_around(x, n):
    """x together with the n nearest floating-point numbers on each side of it."""
    below, above, out = x, x, [x]
    for _ in range(n):
        below = math.nextafter(below, -math.inf)
        above = math.nextafter(above, math.inf)
        out += [below, above]
    return out


@pytest.mark.parametrize("p", [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.5])
def test_contact_points(p):
    """Near the contact points z = |1-p| and z = 1+p, rounding can push arccos and
    sqrt arguments just outside their domain, for example -1.0000000000000002
    this would make math.acos() crash the program, not just give one bad value
    """
    for z0 in (abs(1 - p), 1 + p):
        for z in floats_around(z0, 50):
            assert math.isfinite(flux_uniform(p, z))
            # kappa directly, at ring radii right at the breakpoints |z-p| and z+p
            for r0 in (abs(z - p), z + p):
                for r in floats_around(r0, 3):
                    if 0 < r < 1:
                        assert math.isfinite(kappa(r, p, z))
        for z in floats_around(z0, 2):      # fewer points: the ring sum is slower
            assert math.isfinite(flux_quadratic(p, z, 0.4, 0.25))