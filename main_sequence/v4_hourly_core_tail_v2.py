from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from main_sequence import v4_hourly_core_tail as h

NY = ZoneInfo("America/New_York")


def hour_slug_candidates_2026(start: int) -> list[str]:
    """Resolve 2026 hourly markets without colliding with yearless 2025 slugs.

    2026 hourly Up/Down events use an explicit year in the human-readable slug,
    e.g. bitcoin-up-or-down-june-2-2026-12pm-et.  The old resolver tried the
    yearless slug first; those names already belonged to 2025 markets, so a
    2025 conditionId was paired with a 2026 time window and the Data API
    correctly returned zero trades.  Never accept the ambiguous yearless form.
    """
    d = datetime.fromtimestamp(int(start), tz=timezone.utc).astimezone(NY)
    month = d.strftime("%B").lower()
    hour = d.hour % 12 or 12
    ap = "am" if d.hour < 12 else "pm"
    return [
        f"bitcoin-up-or-down-{month}-{d.day}-{d.year}-{hour}{ap}-et",
        f"btc-updown-1h-{int(start)}",
    ]


h.hour_slug_candidates = hour_slug_candidates_2026
h.PROTOCOL["identity"] = (
    "2026 hourly identity is resolved only from year-explicit human slugs or an exact epoch slug; "
    "ambiguous yearless slugs are forbidden because they collide with 2025 markets."
)
h.PROTOCOL["anti_lookahead"].append(
    "Market identity must correspond to the requested 2026 hourly window; a conditionId from a yearless prior-year slug is forbidden."
)

if __name__ == "__main__":
    h.main()
