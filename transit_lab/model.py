"""Mandel & Agol (2002) transit model."""

def overlap_area(p, z):
    raise NotImplementedError


def flux_uniform(p, z):
    raise NotImplementedError


def flux_quadratic(p, z, u1: float, u2: float):
    raise NotImplementedError


def separation(t, period: float, t0: float, a_over_rstar: float, inc_rad: float):
    raise NotImplementedError


# def transit_duration(period: float, a_over_rstar: float, b: float, p: float) -> float:
#     """Total (first-to-fourth-contact) duration in days. See docs/spec.md 6.3."""
#     raise NotImplementedError
