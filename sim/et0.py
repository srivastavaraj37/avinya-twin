"""FAO-56 Penman-Monteith hourly reference evapotranspiration (ET0).

Reference:
    Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). "Crop
    evapotranspiration - Guidelines for computing crop water requirements."
    FAO Irrigation and Drainage Paper 56.
    - Eq. 53: hourly Penman-Monteith ET0.
    - Eq. 13: slope of saturation vapour pressure curve (Delta).
    - Eq. 7-8: atmospheric pressure and psychrometric constant (gamma).
    - Eq. 39 (and Annex hourly variant): net longwave radiation (Rnl).
    - Eq. 45-46: hourly soil heat flux (G) as a fraction of Rn.
    - Eq. 47: wind speed height adjustment (log-law, z0 for short grass).

Simplification vs. the textbook Rnl formula: FAO-56 Eq. 39 uses the ratio
Rs/Rso (measured vs. clear-sky radiation) as a cloudiness correction, which
is numerically unstable at night (Rso -> 0) and requires computing clear-sky
radiation from solar geometry. Since Open-Meteo already supplies observed
cloud_cover, we instead derive the cloudiness factor directly from it
(``_cloudiness_factor``), which is well-behaved at all hours and is the
"FAO-56 clear-sky formulation with the cloud_cover column" specified for
this project.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sim.psychro import svp_kpa

FloatOrArray = float | npt.NDArray[np.float64]

# --- Physical constants (FAO-56) -------------------------------------------

ALBEDO = 0.23  # dimensionless, FAO-56 reference crop (grass) albedo
# Stefan-Boltzmann constant, hourly form, FAO-56 Box 11: MJ K-4 m-2 h-1
STEFAN_BOLTZMANN_HOURLY = 2.043e-10
W_M2_TO_MJ_M2_PER_HOUR = 0.0036  # 1 W/m2 for 1 hour = 3600 J/m2 = 0.0036 MJ/m2
KELVIN_OFFSET = 273.15  # degC -> K
KELVIN_OFFSET_PSYCHRO = 273.0  # FAO-56 Eq. 53 uses (T + 273) in the wind term

# Wind speed measurement height in the Open-Meteo source data, m.
WIND_MEASUREMENT_HEIGHT_M = 10.0


def wind_speed_2m(u_z: FloatOrArray, z_m: float = WIND_MEASUREMENT_HEIGHT_M) -> FloatOrArray:
    """Convert wind speed measured at height z to the FAO-56 reference height of 2 m.

    FAO-56 Eq. 47 (logarithmic wind profile, short-grass roughness).

    Args:
        u_z: Wind speed at height z_m, m/s.
        z_m: Measurement height, m.

    Returns:
        Wind speed at 2 m, m/s.
    """
    u_z = np.asarray(u_z, dtype=np.float64)
    factor = 4.87 / np.log(67.8 * z_m - 5.42)
    return u_z * factor


def atm_pressure_kpa(elevation_m: float) -> float:
    """Atmospheric pressure from elevation. FAO-56 Eq. 7 (simplified ideal-gas law)."""
    return 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26


def psychrometric_constant(pressure_kpa: float) -> float:
    """Psychrometric constant gamma, kPa/degC. FAO-56 Eq. 8.

    gamma = cp * P / (epsilon * lambda_vap) = 0.000665 * P
    (0.000665 kPa^-1 folds in cp=1.013e-3 MJ/kg/degC, epsilon=0.622,
    lambda_vap=2.45 MJ/kg).
    """
    return 0.000665 * pressure_kpa


def svp_slope_kpa_per_c(t_c: FloatOrArray) -> FloatOrArray:
    """Slope of the saturation vapour pressure curve, Delta. FAO-56 Eq. 13."""
    t_c = np.asarray(t_c, dtype=np.float64)
    es = svp_kpa(t_c)
    return 4098.0 * es / (t_c + 237.3) ** 2


def _cloudiness_factor(cloud_pct: FloatOrArray) -> FloatOrArray:
    """Cloudiness correction factor for net longwave radiation, fcd.

    Plays the role of FAO-56 Eq. 39's (1.35*Rs/Rso - 0.35) term, but derived
    directly from observed cloud cover fraction instead of a clear-sky
    radiation model: fcd = 1.0 at clear sky (cloud=0%), falling toward a
    floor of 0.1 at full overcast (cloud=100%), reflecting that overcast
    skies radiate near-blackbody at cloud-base temperature and strongly
    suppress net longwave loss. Clipped to FAO-56's documented fcd range
    [0.05, 1.0].

    Args:
        cloud_pct: Cloud cover, % (0-100).

    Returns:
        Dimensionless cloudiness factor.
    """
    cloud_frac = np.asarray(cloud_pct, dtype=np.float64) / 100.0
    fcd = 1.0 - 0.9 * cloud_frac
    return np.clip(fcd, 0.05, 1.0)


def net_longwave_mj(t_c: FloatOrArray, rh_pct: FloatOrArray, cloud_pct: FloatOrArray) -> FloatOrArray:
    """Net outgoing longwave radiation, MJ/m2/hr. FAO-56 Eq. 39 (hourly variant).

    Args:
        t_c: Air temperature, degC.
        rh_pct: Relative humidity, % (0-100).
        cloud_pct: Cloud cover, % (0-100).

    Returns:
        Net longwave radiation (positive = energy loss), MJ/m2/hr.
    """
    t_c = np.asarray(t_c, dtype=np.float64)
    ea_kpa = svp_kpa(t_c) * np.asarray(rh_pct, dtype=np.float64) / 100.0
    t_k = t_c + KELVIN_OFFSET
    fcd = _cloudiness_factor(cloud_pct)
    return STEFAN_BOLTZMANN_HOURLY * t_k**4 * (0.34 - 0.14 * np.sqrt(ea_kpa)) * fcd


def net_radiation_mj(
    i_solar_wm2: FloatOrArray,
    t_c: FloatOrArray,
    rh_pct: FloatOrArray,
    cloud_pct: FloatOrArray,
    albedo: float = ALBEDO,
) -> FloatOrArray:
    """Net radiation Rn, MJ/m2/hr, from incoming shortwave and computed Rnl.

    Args:
        i_solar_wm2: Incoming shortwave radiation, W/m2.
        t_c: Air temperature, degC.
        rh_pct: Relative humidity, % (0-100).
        cloud_pct: Cloud cover, % (0-100).
        albedo: Surface albedo, dimensionless.

    Returns:
        Net radiation, MJ/m2/hr.
    """
    rs_mj = np.asarray(i_solar_wm2, dtype=np.float64) * W_M2_TO_MJ_M2_PER_HOUR
    rnl = net_longwave_mj(t_c, rh_pct, cloud_pct)
    return (1.0 - albedo) * rs_mj - rnl


_G_DAY_FRACTION = 0.1  # FAO-56 Eq. 45: G = 0.1*Rn in daylight hours
_G_NIGHT_FRACTION = 0.5  # FAO-56 Eq. 46: G = 0.5*Rn at night
# W/m2 over which the day/night G fraction is smoothly blended, instead of a
# hard step at I_solar=0. FAO-56 defines G as a day/night step function, but
# a hard step makes ET0(I_solar) discontinuous (and locally non-monotonic)
# exactly at sunrise/sunset. Twilight radiation is physically a ramp, not a
# step, so a smoothstep blend over a narrow band is a more faithful (and
# numerically well-behaved) reading of the same FAO-56 rule.
_G_BLEND_WIDTH_WM2 = 50.0


def soil_heat_flux_mj(rn_mj: FloatOrArray, i_solar_wm2: FloatOrArray) -> FloatOrArray:
    """Hourly soil heat flux G from Rn. FAO-56 Eq. 45-46.

    G = 0.1*Rn during daylight hours, 0.5*Rn at night, smoothly blended
    across a narrow twilight band (see module notes) instead of a hard step.
    """
    rn_mj = np.asarray(rn_mj, dtype=np.float64)
    i_solar_wm2 = np.asarray(i_solar_wm2, dtype=np.float64)
    t = np.clip(i_solar_wm2 / _G_BLEND_WIDTH_WM2, 0.0, 1.0)
    smooth_t = t * t * (3.0 - 2.0 * t)  # smoothstep: 0 at t=0, 1 at t=1, zero slope at both ends
    frac = _G_NIGHT_FRACTION + (_G_DAY_FRACTION - _G_NIGHT_FRACTION) * smooth_t
    return frac * rn_mj


def et0_hourly(
    t_c: FloatOrArray,
    rh_pct: FloatOrArray,
    i_solar_wm2: FloatOrArray,
    wind_ms: FloatOrArray,
    cloud_pct: FloatOrArray,
    elevation_m: float = 55.0,
    albedo: float = ALBEDO,
    wind_height_m: float = WIND_MEASUREMENT_HEIGHT_M,
) -> FloatOrArray:
    """FAO-56 Penman-Monteith hourly reference evapotranspiration, ET0.

    Args:
        t_c: Air temperature, degC.
        rh_pct: Relative humidity, % (0-100).
        i_solar_wm2: Incoming shortwave radiation, W/m2.
        wind_ms: Wind speed at wind_height_m, m/s.
        cloud_pct: Cloud cover, % (0-100).
        elevation_m: Site elevation, m above sea level.
        albedo: Surface albedo, dimensionless.
        wind_height_m: Height at which wind_ms was measured, m.

    Returns:
        Reference evapotranspiration, mm/hr (clamped to >= 0).
    """
    t_c = np.asarray(t_c, dtype=np.float64)
    rh_pct = np.asarray(rh_pct, dtype=np.float64)
    i_solar_wm2 = np.asarray(i_solar_wm2, dtype=np.float64)

    es = svp_kpa(t_c)
    ea = es * rh_pct / 100.0
    delta = svp_slope_kpa_per_c(t_c)

    pressure = atm_pressure_kpa(elevation_m)
    gamma = psychrometric_constant(pressure)

    u2 = wind_speed_2m(wind_ms, wind_height_m)

    rn = net_radiation_mj(i_solar_wm2, t_c, rh_pct, cloud_pct, albedo)
    g = soil_heat_flux_mj(rn, i_solar_wm2)

    numerator = 0.408 * delta * (rn - g) + gamma * (37.0 / (t_c + KELVIN_OFFSET_PSYCHRO)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * u2)

    et0 = numerator / denominator
    return np.clip(et0, 0.0, None)


# --- FAO-56 Table 12 tomato crop coefficients -------------------------------

KC_INITIAL = 0.6
KC_DEVELOPMENT = 0.9
KC_MID = 1.15
KC_LATE = 0.80

DAY_INITIAL_END = 29  # last day (0-indexed) of the initial stage
DAY_DEVELOPMENT_END = 69  # last day of the development stage
DAY_MID_END = 119  # last day of the mid-season stage


def kc_for_day(day_index: int) -> float:
    """FAO-56 Table 12 tomato crop coefficient (Kc) for a given day of the crop calendar.

    Stages: initial (0-29) -> development (30-69) -> mid-season (70-119) -> late (120+).

    Args:
        day_index: Days since transplanting/start of simulation, 0-indexed.

    Returns:
        Crop coefficient Kc, dimensionless.
    """
    if day_index <= DAY_INITIAL_END:
        return KC_INITIAL
    if day_index <= DAY_DEVELOPMENT_END:
        return KC_DEVELOPMENT
    if day_index <= DAY_MID_END:
        return KC_MID
    return KC_LATE


def etc_hourly(et0_mm: FloatOrArray, day_index: int) -> FloatOrArray:
    """Crop evapotranspiration ETc = Kc * ET0.

    Args:
        et0_mm: Reference evapotranspiration, mm/hr.
        day_index: Days since start of simulation, 0-indexed (selects Kc).

    Returns:
        Crop evapotranspiration, mm/hr.
    """
    return kc_for_day(day_index) * np.asarray(et0_mm, dtype=np.float64)
