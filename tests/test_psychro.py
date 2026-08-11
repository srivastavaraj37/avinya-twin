"""GATE 2: psychrometric relations sanity checks."""

from __future__ import annotations

import numpy as np

from sim.psychro import abs_humidity, rh_from_chi, svp_kpa, vpd_kpa


def test_abs_humidity_reference_value():
    chi = abs_humidity(20.0, 100.0)
    assert abs(chi - 0.01727) / 0.01727 < 0.01


def test_svp_kpa_at_100c():
    e_s = svp_kpa(100.0)
    assert abs(e_s - 101.3) / 101.3 < 0.01


def test_roundtrip_rh_from_chi():
    for t in range(5, 46, 5):
        for rh in range(10, 101, 10):
            chi = abs_humidity(float(t), float(rh))
            rh_back = rh_from_chi(float(t), chi)
            assert abs(rh_back - rh) < 0.1, f"T={t} RH={rh} -> {rh_back}"


def test_vpd_zero_at_saturation():
    t = np.arange(-10, 51, 1.0)
    vpd = vpd_kpa(t, 100.0)
    assert np.allclose(vpd, 0.0, atol=1e-12)
