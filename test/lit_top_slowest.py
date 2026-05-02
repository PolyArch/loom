#!/usr/bin/env python3
"""Stream lit output while trimming the slow-test summary.

The filter keeps ordinary output moving as soon as it is available, including
lit progress text that is not newline-terminated. Only the final summary blocks
are buffered: the slowest-test block is trimmed to five entries, and the timing
histogram is dropped.
"""

from __future__ import annotations

import re
import sys


SLOWEST = b"Slowest Tests:"
HISTOGRAM = b"Tests Times:"
SPECIALS = (SLOWEST, HISTOGRAM)
TIMED_LINE = re.compile(rb"^[0-9]+\.[0-9]+s:[ \t]")
DIVIDER_LINE = re.compile(rb"^-+\r?\n?$")


def write(data: bytes) -> None:
    if not data:
        return
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def read_line() -> bytes:
    line = bytearray()
    while True:
        ch = sys.stdin.buffer.read(1)
        if not ch:
            break
        line.extend(ch)
        if ch == b"\n":
            break
    return bytes(line)


def line_text(line: bytes) -> bytes:
    return line.strip()


def filter_slowest_block() -> None:
    kept = 0
    while True:
        line = read_line()
        if not line:
            return
        if DIVIDER_LINE.match(line):
            write(line)
            if kept >= 5:
                return
            continue
        if TIMED_LINE.match(line):
            if kept < 5:
                write(line)
                kept += 1
            continue
        write(line)
        if not line.strip():
            return


def filter_histogram_block() -> None:
    while True:
        line = read_line()
        if not line:
            return
        if not line.strip():
            write(line)
            return


def is_prefix_of_special(data: bytes) -> bool:
    return any(special.startswith(data) for special in SPECIALS)


def main() -> int:
    pending = bytearray()
    detecting = True

    while True:
        ch = sys.stdin.buffer.read(1)
        if not ch:
            write(bytes(pending))
            return 0

        if not detecting:
            write(ch)
            if ch == b"\n":
                detecting = True
                pending.clear()
            continue

        pending.extend(ch)
        if ch == b"\n":
            line = bytes(pending)
            stripped = line_text(line)
            if stripped == SLOWEST:
                write(b"Slowest 5 Tests:\n")
                filter_slowest_block()
            elif stripped == HISTOGRAM:
                filter_histogram_block()
            else:
                write(line)
            pending.clear()
            continue

        if is_prefix_of_special(bytes(pending)):
            continue

        write(bytes(pending))
        pending.clear()
        detecting = False


if __name__ == "__main__":
    raise SystemExit(main())
