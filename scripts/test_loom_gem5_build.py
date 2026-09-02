#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("loom_gem5_build.py")
REPO_ROOT = SCRIPT.parents[1]
REPO_TEMP_ROOT = REPO_ROOT / "build" / "test-runs"


def load_helper():
    spec = importlib.util.spec_from_file_location("loom_gem5_build", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def initialize_repository(repository: Path) -> None:
    repository.mkdir(parents=True)
    git(repository, "init", "-q")
    git(repository, "config", "user.name", "Loom Test")
    git(repository, "config", "user.email", "loom-test@example.com")
    git(repository, "config", "commit.gpgsign", "false")


class Gem5BuildHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        REPO_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        cls.module = load_helper()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPO_TEMP_ROOT)
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_bridge_digest_tracks_every_owned_source(self) -> None:
        repository = self.root / "repository"
        bridge = repository / "runtime" / "gem5"
        include = repository / "include" / "Runtime"
        bridge.mkdir(parents=True)
        include.mkdir(parents=True)
        (bridge / "SConscript").write_text("first\n", encoding="utf-8")
        owned_headers = [
            "Gem5BridgeWire.h",
            "Gem5DispatchABI.h",
            "Gem5SpatialBridgeABI.h",
            "SpatialInvocationWire.h",
        ]
        for name in owned_headers:
            (include / name).write_text(f"{name}\n", encoding="utf-8")
        wire = include / owned_headers[0]
        first = self.module.bridge_source_digest(repository)
        wire.write_text("changed\n", encoding="utf-8")
        second = self.module.bridge_source_digest(repository)
        self.assertNotEqual(first, second)

        cache = bridge / "__pycache__"
        cache.mkdir()
        (cache / "generated.cpython-312.pyc").write_bytes(b"generated")
        self.assertEqual(second, self.module.bridge_source_digest(repository))

    def test_readiness_rejects_binary_and_configuration_drift(self) -> None:
        root = self.root / "build"
        binary = root / "build" / "RISCV" / "gem5.opt"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"gem5-binary")
        paths = self.module.BuildPaths(
            root=root,
            source=self.root / "source",
            build_directory=binary.parent,
            binary=binary,
            readiness=root / "loom-gem5-readiness.json",
        )
        expected = {
            "schema": self.module.READINESS_SCHEMA,
            "configuration": {"scons": "4.10.1"},
            "binary": str(binary),
        }
        recorded = dict(expected)
        recorded["binary_sha256"] = self.module.hash_file(binary)
        recorded["version_probe"] = "gem5 version 25"
        paths.readiness.write_text(json.dumps(recorded), encoding="utf-8")
        self.assertEqual(self.module.inspect_readiness(paths, expected), (True, "ready"))

        binary.write_bytes(b"changed")
        self.assertEqual(
            self.module.inspect_readiness(paths, expected)[1],
            "binary digest changed",
        )
        binary.write_bytes(b"gem5-binary")
        drifted = dict(expected)
        drifted["configuration"] = {"scons": "4.10.2"}
        self.assertEqual(
            self.module.inspect_readiness(paths, drifted)[1],
            "source, tool, or build configuration identity changed",
        )

    def test_exact_gitlink_and_clean_checkout_are_required(self) -> None:
        repository = self.root / "repository"
        source = repository / "externals" / "gem5"
        initialize_repository(repository)
        initialize_repository(source)
        (source / "SConstruct").write_text("# pinned\n", encoding="utf-8")
        git(source, "add", "SConstruct")
        git(source, "commit", "-qm", "Pin source")
        pin = git(source, "rev-parse", "HEAD")
        git(
            repository,
            "update-index",
            "--add",
            "--cacheinfo",
            "160000",
            pin,
            "externals/gem5",
        )
        git(repository, "commit", "-qm", "Pin gem5")
        self.assertEqual(
            self.module.validate_gem5_source(repository, source), pin
        )

        (source / "SConstruct").write_text("# changed\n", encoding="utf-8")
        git(source, "add", "SConstruct")
        git(source, "commit", "-qm", "Change source")
        with self.assertRaisesRegex(self.module.BuildError, "does not match gitlink"):
            self.module.validate_gem5_source(repository, source)

        git(source, "reset", "--hard", pin)
        (source / "untracked").write_text("dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(self.module.BuildError, "checkout is dirty"):
            self.module.validate_gem5_source(repository, source)

    def test_linked_worktree_uses_primary_build_and_source(self) -> None:
        repository = self.root / "repository"
        initialize_repository(repository)
        source = repository / "externals" / "gem5"
        source.mkdir(parents=True)
        (source / "SConstruct").write_text("# source\n", encoding="utf-8")
        (repository / "tracked").write_text("root\n", encoding="utf-8")
        git(repository, "add", "tracked")
        git(repository, "commit", "-qm", "Create repository")
        linked = self.root / "linked"
        git(repository, "worktree", "add", "-q", "-b", "linked", str(linked))

        paths = self.module.build_paths(linked)
        self.assertEqual(paths.root, repository / "build" / "gem5")
        self.assertEqual(paths.source, source.resolve())

    def test_build_without_binary_is_rejected(self) -> None:
        root = self.root / "build"
        paths = self.module.BuildPaths(
            root=root,
            source=self.root / "source",
            build_directory=root / "build" / "RISCV",
            binary=root / "build" / "RISCV" / "gem5.opt",
            readiness=root / "readiness.json",
        )
        paths.source.mkdir(parents=True)
        expected = {"configuration": {"scons": {"path": "/usr/bin/true"}}}
        with self.assertRaisesRegex(
            self.module.BuildError, "without producing gem5.opt"
        ):
            self.module.build(self.root, paths, {}, expected, 1)


if __name__ == "__main__":
    unittest.main()
