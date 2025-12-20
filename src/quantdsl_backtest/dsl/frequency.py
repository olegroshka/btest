from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Frequency:
    """A validated bar frequency.

    Design goals:
      - Accept common bar strings like "1d", "5m", "15m", "1h".
      - Keep it extendable (we can later add calendars, business day rules, etc.).
      - Provide a stable canonical string representation.

    Canonical format:
      <positive int><unit>

    Units:
      - m: minutes
      - h: hours
      - d: days
      - w: weeks

    Examples:
      Frequency.parse("1d") == Frequency(1, "d")
      str(Frequency.parse("15m")) == "15m"
    """

    n: int
    unit: str

    _VALID_UNITS = {"m", "h", "d", "w"}

    @staticmethod
    def parse(value: str) -> "Frequency":
        v = (value or "").strip().lower()
        m = re.fullmatch(r"(\d+)([a-z]+)", v)
        if not m:
            raise ValueError(
                f"Invalid frequency {value!r}. Expected formats like '1d', '5m', '15m', '1h'."
            )
        n = int(m.group(1))
        unit = m.group(2)
        if n <= 0:
            raise ValueError(f"Invalid frequency {value!r}: multiplier must be > 0")
        if unit not in Frequency._VALID_UNITS:
            raise ValueError(
                f"Invalid frequency {value!r}: unit must be one of {sorted(Frequency._VALID_UNITS)}"
            )
        return Frequency(n=n, unit=unit)

    def __str__(self) -> str:
        return f"{self.n}{self.unit}"

