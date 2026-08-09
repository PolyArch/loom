#!/usr/bin/env python3

import pathlib
import re
import sys


RUN_LINE = re.compile(r"^# RUN:")
TOOL_INVOCATION = re.compile(r"\b(?:verilator|yosys)\b")
VERSION_QUERY = re.compile(r"\b(?:verilator --version|yosys -V)\b")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_portable_test_parallelism.py TEST_DIRECTORY",
              file=sys.stderr)
        return 2

    root = pathlib.Path(sys.argv[1])
    violations = []
    for path in sorted(root.glob("portable_*.test")):
        invocations = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if (RUN_LINE.match(line) and TOOL_INVOCATION.search(line) and
                    not VERSION_QUERY.search(line)):
                invocations.append(line_number)
        if len(invocations) > 1:
            violations.append((path.name, invocations))

    for path in sorted(root.glob("Portable*Test.cpp")):
        reset_lines = [
            line_number
            for line_number, line in enumerate(path.read_text().splitlines(), 1)
            if "design -reset" in line
        ]
        if reset_lines:
            violations.append((path.name, reset_lines))

    for path, lines in violations:
        rendered = ", ".join(str(line) for line in lines)
        print(f"{path}: independent tool invocations at RUN lines {rendered}",
              file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
