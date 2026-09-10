#!/usr/bin/env python3
"""Filesystem contracts for the destructive scratch/cache cleanup command."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import cleanup


REPOSITORY = Path(__file__).resolve().parent.parent


class CleanupTest(unittest.TestCase):
    def setUp(self):
        scratch = REPOSITORY / "temp"
        scratch.mkdir(exist_ok=True)
        self.tree = tempfile.TemporaryDirectory(prefix="cleanup-test-", dir=scratch)
        self.addCleanup(self.tree.cleanup)
        self.base = Path(self.tree.name)

    def file(self, path, age_days=20):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"retained evidence\x00\xff")
        stamp = time.time() - age_days * cleanup.SECONDS_PER_DAY
        os.utime(path, (stamp, stamp))
        return path

    def run_cleanup(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            result = cleanup.main(list(arguments))
        return result, output.getvalue()

    def test_preview_and_cleanup_preserve_nested_markdown_and_boundaries(self):
        root = self.base / "scratch"
        old = self.file(root / "discard/old.bin")
        guide = self.file(root / "mixed/nested/plan.MD")
        markdown = self.file(root / "root.md")
        hidden_markdown = self.file(root / ".md")
        mixed = self.file(root / "mixed/junk.bin")
        recent = self.file(root / "recent.bin", age_days=0)
        keep = self.file(root / "current/repro.bin")
        external = self.file(self.base / "outside/evidence.bin")
        link = root / "external"
        link.symlink_to(external.parent, target_is_directory=True)
        lock = self.file(root / "cache/locks/active")
        marker = self.file(root / "cache/.loom-external-tool-result-cache")
        protected = [
            guide,
            markdown,
            hidden_markdown,
            recent,
            keep,
            external,
            lock,
            marker,
        ]
        contents = {path: path.read_bytes() for path in protected}
        args = ["--root", str(root), "--keep", str(keep.parent)]
        result, _ = self.run_cleanup(*args)
        self.assertEqual(result, 0)
        self.assertTrue(old.exists() and mixed.exists())
        result, _ = self.run_cleanup(*args, "--apply")
        self.assertEqual(result, 0)
        self.assertFalse(old.parent.exists())
        self.assertFalse(mixed.exists())
        self.assertTrue(link.is_symlink())
        self.assertEqual(contents, {path: path.read_bytes() for path in protected})
        self.assertEqual(cleanup.select_roots(REPOSITORY, [root, guide.parent]), [root])
        with self.assertRaises(ValueError):
            cleanup.select_roots(REPOSITORY, [link])
        with self.assertRaises(ValueError):
            cleanup.select_roots(REPOSITORY, [REPOSITORY])

    def test_oldest_first_across_roots_stops_on_actual_free_space(self):
        first = self.file(self.base / "later-root/z.bin", age_days=30)
        second = self.file(self.base / "earlier-root/a.bin", age_days=20)
        newest = self.file(second.parent / "new.bin", age_days=10)
        target = 1 << 20
        removed = []
        real_remove = cleanup.remove

        def remove(candidate, root):
            removed.append(root.path / candidate.relative)
            return real_remove(candidate, root)

        def usage(_):
            # The first deletion reports no free-space gain. Allocated-byte
            # estimates must not stop actual cleanup before the target is met.
            free = target if not first.exists() and not second.exists() else target - 1
            return SimpleNamespace(total=cleanup.GIB, free=free)

        with (
            mock.patch.object(cleanup.shutil, "disk_usage", side_effect=usage),
            mock.patch.object(cleanup, "remove", side_effect=remove),
        ):
            result, _ = self.run_cleanup(
                "--root",
                str(second.parent),
                "--root",
                str(first.parent),
                "--free-gib",
                str(target / cleanup.GIB),
                "--apply",
            )
        self.assertEqual(result, 0)
        self.assertEqual(removed, [first, second])
        self.assertTrue(newest.exists())

    def test_changed_file_and_replaced_parent_do_not_escape_scan(self):
        root_path = self.base / "scratch"
        changed = self.file(root_path / "changed.bin")
        outside = self.file(self.base / "outside/shared.bin")
        parent = root_path / "parent"
        parent.mkdir()
        # Identical inode/size/mtime on both paths: revalidation alone cannot
        # prevent escape if a parent gets replaced with a directory symlink.
        os.link(outside, parent / outside.name)
        descriptor = os.open(root_path, cleanup.DIRECTORY_FLAGS)
        self.addCleanup(os.close, descriptor)
        root = cleanup.Root(root_path, descriptor, os.fstat(descriptor).st_dev)
        candidates = cleanup.scan([root], [], time.time_ns())
        changed.write_bytes(b"new data")
        parent.rename(root_path / "held")
        parent.symlink_to(outside.parent, target_is_directory=True)
        self.assertTrue(
            all(not cleanup.remove(candidate, root) for candidate in candidates)
        )
        self.assertTrue(outside.exists())
        self.assertEqual(changed.read_bytes(), b"new data")

    def test_hardlinked_old_files_can_both_be_removed(self):
        root = self.base / "scratch"
        original = self.file(root / "first.bin")
        alias = root / "second.bin"
        os.link(original, alias)
        result, _ = self.run_cleanup("--root", str(root), "--apply")
        self.assertEqual(result, 0)
        self.assertFalse(original.exists() or alias.exists())


if __name__ == "__main__":
    unittest.main()
