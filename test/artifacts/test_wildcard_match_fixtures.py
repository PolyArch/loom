#!/usr/bin/env python3

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} REPO_ROOT")

    repo = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(repo / "test" / "artifacts"))
    import wildcard_match_fixtures

    source = repo / "test" / "app" / "wildcard_match" / "main_func.cpp"
    fixture = wildcard_match_fixtures.fixture_from_source("match", source)
    require(fixture.text[10] == ord("A"), "baseline match text should come from C++ source")

    scratch_root = repo / "temp" / "wildcard-match-fixture-test"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_root) as temp_dir:
        copied = Path(temp_dir) / "main_func.cpp"
        shutil.copyfile(source, copied)
        text = copied.read_text()
        text = text.replace("text[10] = 'A';", "text[10] = 'Q';")
        text = text.replace("pattern = {'A', 'B', '?', 'D', 'E', '?', 'G', 'H'};",
                            "pattern = {'Q', 'B', '?', 'D', 'E', '?', 'G', 'H'};")
        copied.write_text(text)

        changed = wildcard_match_fixtures.fixture_from_source("match", copied)
        require(changed.text[10] == ord("Q"), "match text must be parsed from source")
        require(changed.pattern[0] == ord("Q"), "match pattern must be parsed from source")
        require(changed.expected_match == 1, "changed source fixture should still match")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
