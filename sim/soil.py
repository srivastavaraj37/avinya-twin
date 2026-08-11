"""FAO-56 single-layer soil water 'bucket' model (Allen et al. 1998, Ch. 8).

Depletion D (mm) is water missing below field capacity in the root zone.
D=0 means the root zone is at field capacity (as wet as it can hold); D=TAW
means it is at wilting point (all plant-available water gone).

References:
    Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). "Crop
    evapotranspiration - Guidelines for computing crop water requirements."
    FAO Irrigation and Drainage Paper 56. Chapter 8, Eqs. 82-84.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SoilBucket:
    """A single root-zone FAO-56 soil water depletion bucket.

    Args:
        fc: Field capacity, m3/m3 (volumetric water content).
        wp: Wilting point, m3/m3 (volumetric water content).
        zr_m: Effective root depth, m.
        p: FAO-56 depletion fraction for no stress (Table 22), dimensionless.
        d_init_mm: Initial depletion, mm (0 = field capacity).
    """

    fc: float = 0.30  # m3/m3, FAO-56 typical loam field capacity
    wp: float = 0.13  # m3/m3, FAO-56 typical loam wilting point
    zr_m: float = 0.4  # m, effective tomato root depth
    p: float = 0.40  # dimensionless, FAO-56 Table 22 tomato depletion fraction
    d_init_mm: float = 0.0

    def __post_init__(self) -> None:
        # FAO-56 Eq. 82: total available water in the root zone, mm.
        self.taw: float = (self.fc - self.wp) * self.zr_m * 1000.0
        # FAO-56 Eq. 83: readily available water (no-stress threshold), mm.
        self.raw: float = self.p * self.taw
        self.d: float = min(max(self.d_init_mm, 0.0), self.taw)

        # Cumulative accounting so the water balance can be audited exactly:
        #   cum_inflow - cum_et_actual - runoff_mm == d_init_mm - d
        self.runoff_mm: float = 0.0
        self.cum_et_actual_mm: float = 0.0
        self.cum_inflow_mm: float = 0.0
        self._d_init_mm: float = self.d

    def step(self, etc_mm: float, irrigation_mm: float, rain_mm: float) -> float:
        """Advance the bucket by one time step (typically one day or one hour).

        Water is added first (irrigation + rain); any amount beyond what is
        needed to bring the bucket back to field capacity (D=0) is lost as
        runoff/deep percolation and tracked in ``runoff_mm``. Crop ET is then
        subtracted from the remaining water; if there is not enough water in
        the root zone to satisfy the full ETc request, actual ET is capped at
        the water available (D cannot exceed TAW).

        Args:
            etc_mm: Requested crop evapotranspiration this step, mm (>= 0).
            irrigation_mm: Irrigation applied this step, mm (>= 0).
            rain_mm: Rainfall this step, mm (>= 0).

        Returns:
            Updated depletion D, mm.
        """
        inflow = irrigation_mm + rain_mm

        d_after_water = self.d - inflow
        if d_after_water < 0.0:
            runoff_step = -d_after_water
            d_after_water = 0.0
        else:
            runoff_step = 0.0

        d_after_et = d_after_water + etc_mm
        if d_after_et > self.taw:
            actual_et = self.taw - d_after_water
            d_new = self.taw
        else:
            actual_et = etc_mm
            d_new = d_after_et

        self.d = d_new
        self.runoff_mm += runoff_step
        self.cum_et_actual_mm += actual_et
        self.cum_inflow_mm += inflow
        return self.d

    @property
    def stressed(self) -> bool:
        """True when depletion exceeds the readily-available-water threshold."""
        return self.d > self.raw

    @property
    def moisture_pct(self) -> float:
        """Root-zone moisture as a percentage between WP (0%) and FC (100%)."""
        if self.taw <= 0.0:
            return 100.0
        return 100.0 * (1.0 - self.d / self.taw)

    def balance_residual(self) -> float:
        """Water-balance closure residual; should be ~0 within float tolerance.

        cum_inflow - cum_et_actual - runoff_mm - (d_init - d_current)
        """
        return (
            self.cum_inflow_mm
            - self.cum_et_actual_mm
            - self.runoff_mm
            - (self._d_init_mm - self.d)
        )
