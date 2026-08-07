#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = REPO_ROOT / "Makefile"
TEMP_ROOT = REPO_ROOT / "temp"


def run(
    cwd: Path,
    *arguments: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", **(env or {})},
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def git(cwd: Path, *arguments: str) -> str:
    return run(cwd, "git", *arguments).stdout.strip()


class SyncWorktreeMakeTest(unittest.TestCase):
    def setUp(self) -> None:
        TEMP_ROOT.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="sync-worktree-make-test-", dir=TEMP_ROOT
        )
        root = Path(self.temporary.name)
        self.primary = root / "primary"
        self.linked = root / "linked"
        self.log = root / "sync.log"

        run(root, "git", "init", "--initial-branch=main", str(self.primary))
        git(self.primary, "config", "user.name", "Sync Make Test")
        git(self.primary, "config", "user.email", "sync-make@example.com")
        (self.primary / "Makefile").write_text(MAKEFILE.read_text())
        scripts = self.primary / "scripts"
        scripts.mkdir()
        fake = scripts / "sync_branches.py"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            "arguments = sys.argv[1:]\n"
            "with Path(os.environ['SYNC_LOG']).open('a') as stream:\n"
            "    stream.write(' '.join(arguments) + '\\n')\n"
            "if '--dry-run' in arguments and os.environ.get('FAIL_DRY_RUN'):\n"
            "    sys.exit(7)\n"
        )
        git(self.primary, "add", "Makefile", "scripts/sync_branches.py")
        git(self.primary, "commit", "-m", "Add test fixture")
        git(self.primary, "worktree", "add", "-b", "linked", str(self.linked))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def invoke(
        self, worktree: Path, **environment: str
    ) -> subprocess.CompletedProcess[str]:
        return run(
            worktree,
            "make",
            "sync-worktree",
            check=False,
            env={"SYNC_LOG": str(self.log), **environment},
        )

    def test_linked_worktree_runs_preflight_before_real_sync(self) -> None:
        completed = self.invoke(self.linked)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.log.read_text().splitlines(), ["main --dry-run", "main"])

    def test_failed_preflight_prevents_real_sync(self) -> None:
        completed = self.invoke(self.linked, FAIL_DRY_RUN="1")

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(self.log.read_text().splitlines(), ["main --dry-run"])

    def test_primary_worktree_is_rejected_before_sync(self) -> None:
        completed = self.invoke(self.primary)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("linked worktree", completed.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
