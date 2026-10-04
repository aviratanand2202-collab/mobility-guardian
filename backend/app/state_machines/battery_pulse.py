"""
Battery vs. latency pulse logic (spec §2.6).

Under 15% battery while in risk Tier 2 (SUSPICIOUS), the client should not
poll continuously. Instead it wakes the GPS radio for a dense burst, then
sleeps, to balance the ability to compute tortuosity (needs dense
sequential points) against battery preservation for a potential Tier 3
"Last Gasp" SOS transmission.

Tier 3 (CRITICAL) always overrides this - battery lock is dropped entirely
and continuous streaming resumes, down to the "Last Gasp" protocol below
5% battery.

This module computes the *polling instruction* the backend sends to the
client; it does not run on-device. The client (mobile app) is responsible
for actually executing the burst/sleep cycle.
"""
from __future__ import annotations

from dataclasses import dataclass

BATTERY_LOCK_THRESHOLD_PCT = 15
LAST_GASP_THRESHOLD_PCT = 5

PULSE_BURST_SECONDS = 5
PULSE_BURST_HZ = 1
PULSE_SLEEP_SECONDS = 175  # 3 min cycle - 5s burst = 175s sleep
# Worst-case detection delay if a critical escalation begins right after
# a burst ends. Documented limitation, not engineered away (see
# docs/limitations.md) - preserving power for Tier 3 "Last Gasp" SOS is an
# intentional tradeoff.
WORST_CASE_DETECTION_DELAY_SEC = PULSE_SLEEP_SECONDS


@dataclass
class PollingInstruction:
    mode: str  # "CONTINUOUS" | "PULSE" | "LAST_GASP"
    burst_seconds: int | None = None
    burst_hz: int | None = None
    sleep_seconds: int | None = None


def compute_polling_instruction(
    risk_tier: str, battery_pct: int
) -> PollingInstruction:
    """
    risk_tier: one of QUIESCENT / NORMAL_TRANSIT / SUSPICIOUS / CRITICAL
    battery_pct: 0-100

    Tier 3 (CRITICAL) always overrides the battery lock - safety beats
    hardware preservation.
    """
    if risk_tier == "CRITICAL":
        if battery_pct < LAST_GASP_THRESHOLD_PCT:
            # Last Gasp: suspend streaming, force one high-accuracy fix,
            # send a single priority SMS payload, shut down non-essential
            # background services. Orchestrated by the caller - this
            # function just signals the mode.
            return PollingInstruction(mode="LAST_GASP")
        return PollingInstruction(mode="CONTINUOUS")

    if risk_tier == "SUSPICIOUS" and battery_pct < BATTERY_LOCK_THRESHOLD_PCT:
        return PollingInstruction(
            mode="PULSE",
            burst_seconds=PULSE_BURST_SECONDS,
            burst_hz=PULSE_BURST_HZ,
            sleep_seconds=PULSE_SLEEP_SECONDS,
        )

    # Normal tiers / adequate battery: handled by the base risk-adaptive
    # polling table (Tier 0-2 intervals), not this module.
    return PollingInstruction(mode="CONTINUOUS")
