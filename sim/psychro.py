"""Psychrometric relations: saturation vapour pressure, humidity, dewpoint.

All functions are vectorised (they accept Python floats or numpy arrays and
use only numpy ufuncs, so broadcasting/array inputs work transparently).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatOrArray = float | npt.NDArray[np.float64]

# Tetens (1930) saturation vapour pressure formula, as adopted by FAO-56
# (Allen et al. 1998, Eq. 11), coefficients tuned for T in degC, e_s in kPa.
_TETENS_A = 0.6108  # kPa
_TETENS_B = 17.27  # dimensionless
_TETENS_C = 237.3  # degC

# Specific gas constant for water vapour: R_universal / M_water
# = 8314.47 J/(kmol.K) / 18.015 kg/kmol = 461.5 J/(kg.K)
R_V = 461.5  # J/(kg.K)

_KELVIN_OFFSET = 273.15  # degC -> K


def svp_kpa(t_c: FloatOrArray) -> FloatOrArray:
    """Saturation vapour pressure via the Tetens/FAO-56 formula.

    Args:
        t_c: Air temperature, degC.

    Returns:
        Saturation vapour pressure, kPa.
    """
    t_c = np.asarray(t_c, dtype=np.float64)
    result = _TETENS_A * np.exp(_TETENS_B * t_c / (t_c + _TETENS_C))
    return result


def vpd_kpa(t_c: FloatOrArray, rh_pct: FloatOrArray) -> FloatOrArray:
    """Vapour pressure deficit: e_s(T) * (1 - RH/100).

    Args:
        t_c: Air temperature, degC.
        rh_pct: Relative humidity, % (0-100).

    Returns:
        Vapour pressure deficit, kPa.
    """
    return svp_kpa(t_c) * (1.0 - np.asarray(rh_pct, dtype=np.float64) / 100.0)


def abs_humidity(t_c: FloatOrArray, rh_pct: FloatOrArray) -> FloatOrArray:
    """Absolute humidity (vapour density) from temperature and RH.

    Uses the ideal gas law for water vapour: rho_v = e / (R_v * T_K),
    with e the actual (not saturation) vapour pressure.

    Args:
        t_c: Air temperature, degC.
        rh_pct: Relative humidity, % (0-100).

    Returns:
        Absolute humidity, kg water vapour per m3 moist air.
    """
    t_c = np.asarray(t_c, dtype=np.float64)
    e_kpa = svp_kpa(t_c) * np.asarray(rh_pct, dtype=np.float64) / 100.0
    e_pa = e_kpa * 1000.0
    t_k = t_c + _KELVIN_OFFSET
    return e_pa / (R_V * t_k)


def chi_sat(t_c: FloatOrArray) -> FloatOrArray:
    """Saturation absolute humidity (RH=100%).

    Args:
        t_c: Air temperature, degC.

    Returns:
        Saturation absolute humidity, kg water vapour per m3 moist air.
    """
    return abs_humidity(t_c, 100.0)


def rh_from_chi(t_c: FloatOrArray, chi: FloatOrArray) -> FloatOrArray:
    """Relative humidity from temperature and absolute humidity (inverse of abs_humidity).

    Args:
        t_c: Air temperature, degC.
        chi: Absolute humidity, kg water vapour per m3 moist air.

    Returns:
        Relative humidity, % (clipped to [0, 100]).
    """
    t_c = np.asarray(t_c, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)
    t_k = t_c + _KELVIN_OFFSET
    e_pa = chi * R_V * t_k
    e_kpa = e_pa / 1000.0
    rh = 100.0 * e_kpa / svp_kpa(t_c)
    return np.clip(rh, 0.0, 100.0)


def dewpoint_c(t_c: FloatOrArray, rh_pct: FloatOrArray) -> FloatOrArray:
    """Dewpoint temperature, inverting the Tetens formula at actual vapour pressure.

    Args:
        t_c: Air temperature, degC.
        rh_pct: Relative humidity, % (0-100).

    Returns:
        Dewpoint temperature, degC.
    """
    t_c = np.asarray(t_c, dtype=np.float64)
    rh_pct = np.asarray(rh_pct, dtype=np.float64)
    e_kpa = svp_kpa(t_c) * rh_pct / 100.0
    # Guard against log(0) at RH=0; dewpoint is unbounded below in that limit.
    e_kpa = np.clip(e_kpa, 1e-6, None)
    ln_ratio = np.log(e_kpa / _TETENS_A)
    return _TETENS_C * ln_ratio / (_TETENS_B - ln_ratio)
