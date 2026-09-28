"""Mandel & Agol (2002) transit model."""

import math

def flux_uniform(p, z):
    if z >= 1 + p: return 1.0        # no overlap
    if z <= 1 - p: return 1 - p**2      # planet fully inside the star
    if z <= p - 1: return 0.0         # star fully covered
    clamp = lambda x: max(-1.0, min(1.0, x))
    alpha = math.acos(clamp((1 - p**2 + z**2) / (2*z)))
    beta = math.acos(clamp((p**2 + z**2 - 1) / (2*p*z)))
    sqrt = math.sqrt(max(0.0, 4*z**2 - (1 + z**2 - p**2)**2))
    return 1 - (p**2 * beta + alpha - 0.5 * sqrt) / math.pi


def kappa(r, p, z):
    # helper for quadratic limb darkening
    if r <= p - z: return math.pi
    if abs(z - p) < r < z + p:
        return math.acos(max(-1.0, min(1.0, (r**2 + z**2 - p**2) / (2*r*z))))
    return 0

def flux_quadratic(p, z, u1: float, u2: float, N=4000):
    dr = 1.0 / N
    blocked = total = 0.0
    for i in range(N):
        r  = (i + 0.5) * dr                       # middle of ring i
        mu = math.sqrt(1 - r**2)
        I  = 1 - u1*(1 - mu) - u2*(1 - mu)**2
        total   += I * 2*math.pi*r * dr
        blocked += I * 2*r * kappa(r, p, z) * dr  # the 3-case function
    return 1 - blocked / total


def separation(t, period: float, t0: float, a_over_rstar: float, inc_rad: float):
    phase = 2 * math.pi * (t - t0) / period          # omega * (t - t0)
    if math.cos(phase) <= 0:
        return math.inf                              # planet behind the star
    return a_over_rstar * math.sqrt(
        math.sin(phase)**2 + (math.cos(phase) * math.cos(inc_rad))**2)


# def transit_duration(period: float, a_over_rstar: float, b: float, p: float) -> float:
#     raise NotImplementedError
