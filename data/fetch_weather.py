"""Fetch and cache hourly historical weather for Guwahati from Open-Meteo.

Open-Meteo's "archive" API (ERA5-based reanalysis) is free and requires no
API key: https://archive-api.open-meteo.com/v1/archive

Run as a script to (re)build the full multi-year cache, one CSV per calendar
year (``data/guwahati_<year>.csv``), spanning 2021-01-01 through the most
recent complete day the archive has (see ``latest_complete_day``):

    python data/fetch_weather.py

For a single custom range, call ``fetch_weather(start_date, end_date)``
directly; for the full multi-year set, call ``fetch_weather_multi_year()``.
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.config import default_config  # noqa: E402

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
]

RENAME_MAP = {
    "temperature_2m": "T_out",
    "relative_humidity_2m": "RH_out",
    "shortwave_radiation": "I_solar",
    "wind_speed_10m": "wind",
    "precipitation": "rain",
    "cloud_cover": "cloud",
}

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = DATA_DIR / "guwahati_2025.csv"

# Longest gap (hours) we are willing to fill by linear interpolation before
# treating missing data as a hard failure.
MAX_INTERP_GAP_HOURS = 2

# Multi-year cache range. Open-Meteo's archive API blends recent days with
# short-range forecast/analysis data that is not yet final ERA5 reanalysis
# (there is no hard published cutoff), so rather than trust "no nulls today"
# at face value we stay a conservative ARCHIVE_LAG_DAYS behind the wall clock
# for the current year's endpoint -- matched to the ~10-day lag the project
# spec assumes.
MULTI_YEAR_START_YEAR = 2021
ARCHIVE_LAG_DAYS = 10


def latest_complete_day(today: _dt.date | None = None) -> str:
    """Most recent day we treat as reliably-final archive data, ISO string."""
    today = today or _dt.date.today()
    return (today - _dt.timedelta(days=ARCHIVE_LAG_DAYS)).isoformat()


def year_csv_path(year: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"guwahati_{year}.csv"


def _expected_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Hourly timestamp index spanning [start_date 00:00, end_date 23:00]."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(hours=23)
    return pd.date_range(start, end, freq="h")


def _covers_range(df: pd.DataFrame, start_date: str, end_date: str) -> bool:
    if df.empty:
        return False
    expected = _expected_index(start_date, end_date)
    return expected[0] >= df.index.min() and expected[-1] <= df.index.max()


def _fetch_from_api(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str,
) -> pd.DataFrame:
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "hourly": ",".join(HOURLY_VARS),
        # Open-Meteo's default wind speed unit is km/h; request m/s explicitly
        # so wind_speed_10m lands directly in the SI unit our physics uses.
        "wind_speed_unit": "ms",
    }
    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "Failed to fetch weather data from Open-Meteo archive API "
            f"({ARCHIVE_URL}). Check your network connection and try again. "
            f"Underlying error: {exc}"
        ) from exc

    payload = resp.json()
    if "hourly" not in payload:
        raise RuntimeError(
            f"Open-Meteo response missing 'hourly' block. Full response: {payload}"
        )

    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").rename(columns=RENAME_MAP)
    df = df[list(RENAME_MAP.values())]
    return df


# Forecast mode ("opt-in, next N days" -- Live Simulation page only). Kept
# separate from fetch_weather()/fetch_weather_multi_year() above, which stay
# archive-only and untouched: this is the one path in the whole project that
# reads live, not-yet-validated Open-Meteo forecast data, so it is never the
# default and never mixed into the cached multi-year record those other
# functions build.
def fetch_open_meteo_forecast(
    latitude: float,
    longitude: float,
    timezone: str,
    forecast_days: int = 12,
    timeout_s: float = 15.0,
) -> pd.DataFrame:
    """Fetch the next ``forecast_days`` days of hourly forecast weather.

    Reuses HOURLY_VARS/RENAME_MAP so the returned frame has exactly the same
    six columns (T_out, RH_out, I_solar, wind, rain, cloud) as the cached
    archive data -- everything downstream of this call (sim.engine.run, the
    controllers) sees an identical schema regardless of which source the
    weather came from.

    Raises RuntimeError on any network failure, timeout, or malformed
    response -- callers are expected to catch this and fall back to cached
    mode rather than let it propagate to a crash.
    """
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "forecast_days": forecast_days,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=timeout_s)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch forecast weather from Open-Meteo ({FORECAST_URL}). "
            f"Underlying error: {exc}"
        ) from exc

    payload = resp.json()
    if "hourly" not in payload:
        raise RuntimeError(f"Open-Meteo forecast response missing 'hourly' block. Full response: {payload}")

    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").rename(columns=RENAME_MAP)
    df = df[list(RENAME_MAP.values())]
    if df.isna().any().any():
        # Forecast data can have short leading/trailing NaN runs (e.g. a
        # variable not yet initialized for the last forecast hour); reuse
        # the same conservative interpolation rule as the archive path
        # rather than inventing a separate tolerance.
        df = _interpolate_short_gaps(df)
    return df


def _interpolate_short_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate NaN runs of length <= MAX_INTERP_GAP_HOURS.

    Raises:
        ValueError: if any column has a NaN run longer than the threshold.
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        is_na = s.isna()
        if not is_na.any():
            continue
        # Identify contiguous NaN run lengths.
        group_id = (is_na != is_na.shift()).cumsum()
        run_lengths = is_na.groupby(group_id).transform("sum")
        max_bad_run = run_lengths[is_na].max() if is_na.any() else 0
        if max_bad_run > MAX_INTERP_GAP_HOURS:
            raise ValueError(
                f"Column '{col}' has a missing-data gap of {int(max_bad_run)} "
                f"hours, exceeding the {MAX_INTERP_GAP_HOURS}-hour interpolation "
                "limit. Refusing to silently fabricate data -- refetch or "
                "inspect the source."
            )
        df[col] = s.interpolate(method="linear", limit_direction="both")
    return df


def fetch_weather(
    start_date: str | None = None,
    end_date: str | None = None,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    force_refetch: bool = False,
) -> pd.DataFrame:
    """Fetch (or load cached) hourly Guwahati weather, cleaned and gap-filled.

    Args:
        start_date: ISO date string, e.g. "2025-03-01". Defaults to config.yaml.
        end_date: ISO date string, e.g. "2025-08-31". Defaults to config.yaml.
        csv_path: Where to cache/read the CSV.
        force_refetch: If True, always hit the API even if a cache exists.

    Returns:
        DataFrame indexed by hourly local timestamp (tz-naive, Asia/Kolkata
        clock time) with columns: T_out (degC), RH_out (%), I_solar (W/m2),
        wind (m/s), rain (mm), cloud (%). No NaNs; short gaps interpolated.
    """
    cfg = default_config()
    start_date = start_date or cfg["weather"]["start_date"]
    end_date = end_date or cfg["weather"]["end_date"]
    latitude = cfg["location"]["latitude"]
    longitude = cfg["location"]["longitude"]
    timezone = cfg["location"]["timezone"]

    csv_path = Path(csv_path)

    cached: pd.DataFrame | None = None
    if csv_path.exists() and not force_refetch:
        cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if _covers_range(cached, start_date, end_date):
            df = cached.loc[
                pd.Timestamp(start_date) : pd.Timestamp(end_date)
                + pd.Timedelta(hours=23)
            ]
            return _interpolate_short_gaps(df)

    df = _fetch_from_api(latitude, longitude, start_date, end_date, timezone)
    df = _interpolate_short_gaps(df)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)
    return df


def fetch_weather_multi_year(
    start_year: int = MULTI_YEAR_START_YEAR,
    end_date: str | None = None,
    data_dir: Path = DATA_DIR,
    force_refetch: bool = False,
) -> pd.DataFrame:
    """Fetch/load the full multi-year record, one cached CSV per calendar year.

    Each year is fetched (or read from cache) independently via
    ``fetch_weather``, so a re-run only hits the network for years whose
    cache doesn't yet cover the requested range -- in particular the current
    (partial) year's cache grows automatically as ``end_date`` advances on
    later runs, without re-fetching already-complete prior years.

    Args:
        start_year: First calendar year to include (Jan 1).
        end_date: Last day to include, ISO string. Defaults to
            ``latest_complete_day()``.
        data_dir: Directory holding/receiving the per-year CSVs.
        force_refetch: If True, re-fetch every year even if cached.

    Returns:
        A single DataFrame, hourly-indexed, contiguous, spanning
        start_year-01-01 through end_date, concatenated from the per-year
        caches.
    """
    end_date = end_date or latest_complete_day()
    end_year = pd.Timestamp(end_date).year
    if end_year < start_year:
        raise ValueError(f"end_date {end_date} is before start_year {start_year}")

    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        y_start = f"{year}-01-01"
        y_end = f"{year}-12-31" if year < end_year else end_date
        csv_path = year_csv_path(year, data_dir)
        print(f"  year {year}: {y_start}..{y_end} -> {csv_path.name}")
        frames.append(
            fetch_weather(y_start, y_end, csv_path=csv_path, force_refetch=force_refetch)
        )

    df = pd.concat(frames).sort_index()
    dupes = int(df.index.duplicated().sum())
    if dupes:
        raise ValueError(f"Multi-year concat produced {dupes} duplicate timestamps")
    expected_hours = len(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    if len(df) != expected_hours:
        raise ValueError(
            f"Multi-year concat is not contiguous: {len(df)} rows, expected {expected_hours} "
            "(gap between year files?)"
        )
    return df


def load_multi_year_weather(
    start_year: int = MULTI_YEAR_START_YEAR,
    end_date: str | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Load the cached multi-year record without touching the network.

    Assumes ``fetch_weather_multi_year`` has already been run at least once
    for the requested range (each year's cache satisfies ``_covers_range``);
    raises if a year's CSV is missing.
    """
    end_date = end_date or latest_complete_day()
    end_year = pd.Timestamp(end_date).year
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        csv_path = year_csv_path(year, data_dir)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found -- run fetch_weather_multi_year() or "
                "python data/fetch_weather.py first"
            )
        y_start = f"{year}-01-01"
        y_end = f"{year}-12-31" if year < end_year else end_date
        cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        frames.append(_interpolate_short_gaps(cached.loc[y_start : pd.Timestamp(y_end) + pd.Timedelta(hours=23)]))
    return pd.concat(frames).sort_index()


def validate_gate1(df: pd.DataFrame) -> dict[str, Any]:
    """Compute Phase-1 GATE-1 diagnostics. Raises AssertionError on failure."""
    results: dict[str, Any] = {}

    n_nan = int(df.isna().sum().sum())
    results["no_nans"] = n_nan == 0
    assert n_nan == 0, f"Found {n_nan} NaNs after ingestion"

    expected_hours = len(
        pd.date_range(df.index.min(), df.index.max(), freq="h")
    )
    results["row_count"] = len(df)
    results["expected_hours"] = expected_hours
    assert len(df) == expected_hours, (
        f"Row count {len(df)} != expected {expected_hours} hours"
    )

    jja = df[(df.index.month >= 6) & (df.index.month <= 8)]
    mean_rh = float(jja["RH_out"].mean())
    results["jja_mean_rh"] = mean_rh
    assert 78 <= mean_rh <= 95, (
        f"June-Aug mean RH_out={mean_rh:.1f}% outside [78, 95] monsoon band"
    )

    night = df[(df.index.hour >= 20) | (df.index.hour <= 4)]
    max_night_solar = float(night["I_solar"].max())
    results["max_night_solar"] = max_night_solar
    assert max_night_solar == 0, (
        f"I_solar nonzero at night: max={max_night_solar}"
    )

    june = df[df.index.month == 6]
    daily_max = june.groupby(june.index.date)["I_solar"].max()
    results["june_daily_max_solar_min"] = float(daily_max.min())
    results["june_daily_max_solar_max"] = float(daily_max.max())
    assert daily_max.between(500, 1000).all(), (
        f"June daily max I_solar out of [500,1000]: "
        f"min={daily_max.min()}, max={daily_max.max()}"
    )

    return results


def validate_gate1_multi_year(df: pd.DataFrame) -> dict[str, Any]:
    """Multi-year GATE-1 diagnostics: per-year checks, not pooled.

    ``validate_gate1`` (above) is left completely unmodified and still applies
    unchanged to any single-year (or single-window) frame -- e.g. calling it
    on just the 2025 slice reproduces the exact original Gate 1 result. This
    function is a *separate*, additionally-scoped check for a multi-year
    sample, because two of the original checks don't generalize correctly by
    simply pooling more years into the same bounds:

    - JJA mean RH_out: checked per-year (not pooled), since one anomalous
      year averaging against five normal ones could hide a real problem.
    - June daily max I_solar: the original [500, 1000] W/m2 band was
      calibrated on a single year (2025) and is not a physical bound -- a
      100%-overcast monsoon-onset deluge day is real weather, not a bug.
      Confirmed here: 2022-06-13 has a June daily max of just 184 W/m2, with
      cloud_cover=100% and hourly rain up to 11mm on that day (checked
      directly against the fetched data). We replace the tight band with a
      physically-derived one instead of silently loosening the original:
      floor of 50 W/m2 (a true "sensor stuck near zero" bug would show 0-ish
      for the *entire* day, not just a low peak -- 50 is comfortably above
      that failure mode while permitting genuine deep-overcast days) and a
      ceiling of 1050 W/m2 (clear-sky extraterrestrial max at this latitude
      is ~950-1000 W/m2; anything past ~1050 signals a unit/scale bug, e.g.
      a stray W/m2->kW/m2 conversion). Checked per-year so one bad year can't
      hide inside a pooled min/max.

    The no-NaN, row-count, and night-solar-must-be-zero checks generalize to
    any sample size without modification and are simply re-run here directly
    (not re-derived) against the full multi-year frame.
    """
    results: dict[str, Any] = {}

    n_nan = int(df.isna().sum().sum())
    results["no_nans"] = n_nan == 0
    assert n_nan == 0, f"Found {n_nan} NaNs after ingestion"

    expected_hours = len(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    results["row_count"] = len(df)
    results["expected_hours"] = expected_hours
    assert len(df) == expected_hours, f"Row count {len(df)} != expected {expected_hours} hours"

    night = df[(df.index.hour >= 20) | (df.index.hour <= 4)]
    max_night_solar = float(night["I_solar"].max())
    results["max_night_solar"] = max_night_solar
    assert max_night_solar == 0, f"I_solar nonzero at night: max={max_night_solar}"

    per_year_jja: dict[int, float] = {}
    per_year_june_solar: dict[int, tuple[float, float]] = {}
    for year, group in df.groupby(df.index.year):
        jja = group[(group.index.month >= 6) & (group.index.month <= 8)]
        if not jja.empty:
            mean_rh = float(jja["RH_out"].mean())
            per_year_jja[int(year)] = mean_rh
            assert 78 <= mean_rh <= 95, (
                f"{year} June-Aug mean RH_out={mean_rh:.1f}% outside [78, 95] monsoon band"
            )

        june = group[group.index.month == 6]
        if not june.empty:
            daily_max = june.groupby(june.index.date)["I_solar"].max()
            lo, hi = float(daily_max.min()), float(daily_max.max())
            per_year_june_solar[int(year)] = (lo, hi)
            assert 50 <= lo and hi <= 1050, (
                f"{year} June daily max I_solar out of [50, 1050]: min={lo}, max={hi}"
            )
    results["per_year_jja_mean_rh"] = per_year_jja
    results["per_year_june_daily_max_solar"] = per_year_june_solar

    return results


if __name__ == "__main__":
    print(f"Fetching multi-year record: {MULTI_YEAR_START_YEAR}-01-01 .. {latest_complete_day()}")
    weather = fetch_weather_multi_year()
    print(f"\nFetched/loaded {len(weather)} hourly rows across "
          f"{weather.index.year.nunique()} years -> {DATA_DIR}/guwahati_<year>.csv")
    print(weather.describe())

    gate = validate_gate1_multi_year(weather)
    print("\nGATE 1 (multi-year) results:")
    for k, v in gate.items():
        print(f"  {k}: {v}")

    print("\nGATE 1: PASS")
