"""Hardware bill of materials for retrofitting the avinya-twin controller
onto an existing 100 m2 polyhouse.

Every price below is a real single-unit Indian retail listing found via web
search (August 2026), not an invented or textbook number -- each line
carries the specific source it came from. Prices are indicative retail
prices for one unit at the time of research; they exclude GST where the
listing itself didn't state GST-inclusive, exclude wiring/cabling/mounting
hardware and labour, and will drift over time and across sellers the way
any component-market price does. Treat this as an order-of-magnitude retrofit
budget, not a quote.

Component selection matches config.yaml's already-documented actuator specs
where one exists (e.g. the two 125 W HAF fans -- see config.yaml's `fan:`
section) so the BOM prices the system this project actually models, not a
generic IoT kit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BOMItem:
    component: str
    role: str
    qty: int
    unit_price_inr: float
    source_name: str
    source_url: str
    note: str = ""

    @property
    def line_total_inr(self) -> float:
        return self.qty * self.unit_price_inr


BOM: list[BOMItem] = [
    BOMItem(
        component="ESP32-S3 dev board",
        role="Edge controller (runs vent/threshold/fan logic on-device)",
        qty=1,
        unit_price_inr=899.0,
        source_name="Robocraze (Seeed Studio XIAO ESP32-S3)",
        source_url="https://robocraze.com/products/seeed-studio-xiao-esp32-s3-development-board-supports-wi-fi-bluetooth-5-0",
    ),
    BOMItem(
        component="DHT22 temperature/humidity sensor",
        role="Indoor T_in/RH_in sensing (x2 for redundancy/averaging)",
        qty=2,
        unit_price_inr=300.0,
        source_name="IndiaMART listing (Vishal Electronics, Bengaluru)",
        source_url="https://www.indiamart.com/proddetail/dht22-am2302-digital-temperature-humidity-sensor-22317482788.html",
    ),
    BOMItem(
        component="Capacitive soil moisture sensor V1.2",
        role="Soil depletion sensing (x3, spatial averaging across the 100 m2 bed)",
        qty=3,
        unit_price_inr=83.0,
        source_name="Robokits India",
        source_url="https://robokits.co.in/sensors/water-moisture/capacitive-soil-moisture-sensor-v1.2",
    ),
    BOMItem(
        component="BH1750 ambient light sensor",
        role="I_solar proxy (digital I2C lux sensor)",
        qty=1,
        unit_price_inr=132.0,
        source_name="Robokits India",
        source_url="https://robokits.co.in/sensors/light-sensor/bh1750-light-intensity-module",
    ),
    BOMItem(
        component="4-channel 5V relay module",
        role="Switches fan(s), solenoid valve, and actuator direction relays",
        qty=1,
        unit_price_inr=170.0,
        source_name="IndiaMART listing (Engineers Bazaar)",
        source_url="https://www.indiamart.com/proddetail/4-channel-5v-relay-board-19179450812.html",
    ),
    BOMItem(
        component="12V linear actuator, 300mm stroke, 1500N",
        role="Vent opening mechanism (drives vent_frac 0-1 as stroke position)",
        qty=1,
        unit_price_inr=3599.0,
        source_name="IndiaMART listing (New Delhi)",
        source_url="https://www.indiamart.com/proddetail/12v-300mm-stroke-length-linear-actuator-15mm-s-1500n-24604906597.html",
    ),
    BOMItem(
        component="HAF circulation fan",
        role="Leaf boundary-layer fan (matches config.yaml's 125W-per-fan spec)",
        qty=2,
        unit_price_inr=1950.0,
        source_name="IndiaMART listing, 18-inch metal circulation fan (Indore)",
        source_url="https://www.indiamart.com/proddetail/18-inch-exhaust-fan-18498647191.html",
        note="Closest publicly-priced analog to a small 125W HAF fan; large 1.5HP+ "
        "'greenhouse exhaust fan' listings (Rs 18,000-25,000) are a different, "
        "much higher-power product class than the config.yaml spec models.",
    ),
    BOMItem(
        component="12V DC solenoid valve, 1/2 inch",
        role="Irrigation actuation",
        qty=1,
        unit_price_inr=500.0,
        source_name="Flipkart (Parijata Industrial)",
        source_url="https://www.flipkart.com/parijata-industrial-water-solenoid-valve-12v-dc-500ma-1-2-x1-2-commercial-purifier-automatic-control-valves/p/itmd887f24fd6674",
    ),
    BOMItem(
        component="IP65 enclosure box",
        role="Weatherproofs the controller/relay/SMPS assembly",
        qty=1,
        unit_price_inr=295.0,
        source_name="TradeIndia listing (Bhadra Enclosure Systems, Mumbai)",
        source_url="https://www.tradeindia.com/products/abs-enclosures-c3969321.html",
    ),
    BOMItem(
        component="12V 5A SMPS power supply",
        role="Powers the controller, sensors, relays, and actuator",
        qty=1,
        unit_price_inr=579.0,
        source_name="Indian Hobby Center",
        source_url="https://www.indianhobbycenter.com/products/12v-5a-smps-power-suply",
    ),
]


def bom_total_inr() -> float:
    return sum(item.line_total_inr for item in BOM)
