#!/usr/bin/env python3
"""Regenerate the audited ExpNegativeQ64Table_1_0 C++ constants.

The checked-in C++ table is the protocol authority. This offline utility is
not invoked by the build or runtime; its independent precision/digest checks
only make deliberate table maintenance reproducible.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_FLOOR, localcontext
import hashlib
from pathlib import Path
import struct


EXPECTED_COUNT = 11356
EXPECTED_DIGEST = "88a35fea368b5df890aa790239ca681154f69541c3e7dab05cf60dbc3890bfbf"


def thresholds(precision: int) -> list[int]:
    with localcontext() as context:
        context.prec = precision
        scale = Decimal(2) ** 64
        result: list[int] = []
        index = 1
        while True:
            value = (scale * (-(Decimal(index) / Decimal(256))).exp()).to_integral_value(
                rounding=ROUND_FLOOR
            )
            threshold = int(value)
            if threshold == 0:
                return result
            result.append(threshold)
            index += 1


def render(values: list[int]) -> str:
    lines = [
        '#include "PnR/DeterministicSearchProtocol.h"',
        "",
        "#include <array>",
        "#include <cassert>",
        "#include <cstdint>",
        "",
        "namespace {",
        "",
        "// clang-format off",
        f"constexpr std::array<std::uint64_t, {len(values)}> thresholds{{{{",
    ]
    for offset in range(0, len(values), 12):
        row = ", ".join(f"UINT64_C(0x{value:016x})" for value in values[offset : offset + 12])
        lines.append(f"  {row},")
    lines.extend(
        [
            "}};",
            "// clang-format on",
            "",
            "} // namespace",
            "",
            "llvm::ArrayRef<std::uint64_t> loom::pnr::expNegativeQ64Thresholds() {",
            "  return thresholds;",
            "}",
            "",
            "std::uint64_t loom::pnr::expNegativeQ64Threshold(std::uint64_t ratioIndex) {",
            "  assert(ratioIndex != 0 && \"acceptance ratio index must be positive\");",
            "  if (ratioIndex > thresholds.size())",
            "    return 0;",
            "  return thresholds[ratioIndex - 1];",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    values = thresholds(220)
    if values != thresholds(320):
        raise SystemExit("precision audit produced different tables")
    payload = b"".join(struct.pack(">Q", value) for value in values)
    digest = hashlib.sha256(payload).hexdigest()
    if len(values) != EXPECTED_COUNT or digest != EXPECTED_DIGEST:
        raise SystemExit(f"table audit failed: count={len(values)} digest={digest}")
    arguments.output.write_text(render(values), encoding="ascii")


if __name__ == "__main__":
    main()
