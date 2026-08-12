#!/usr/bin/env python3

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from resolve_experiment_root import (  # noqa: E402
    EXTERNAL_TOOL_CACHE_MARKER,
    EXTERNAL_TOOL_CACHE_MARKER_CONTENTS,
    EXTERNAL_TOOL_CACHE_MEMBERS,
    EXTERNAL_TOOL_CACHE_ROOT_ENVIRONMENT,
)


CACHE_ENTRY_SCHEMA = "loom.external_tool_command_cache_entry"
CACHE_ENTRY_VERSION = "1.0"
COMMAND_ENTRY_DIRECTORY = "command-entries"
COMMAND_LOCK_DIRECTORY = "command-locks"
REAL_PATH_ENVIRONMENT = "LOOM_HARDWARE_TOOL_REAL_PATH"
WORK_ROOT_ENVIRONMENT = "LOOM_HARDWARE_TOOL_WORK_ROOT"
SUPPORTED_TOOLS = frozenset(
    {
        "quartus_asm",
        "quartus_fit",
        "quartus_sh",
        "quartus_sta",
        "quartus_syn",
        "verilator",
        "vivado",
        "yosys",
    }
)
CONFIGURATION_ENVIRONMENT = (
    "CFLAGS",
    "CPPFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "MAKEFLAGS",
)
VERSION_ARGUMENTS = {
    "quartus_asm": ("--version",),
    "quartus_fit": ("--version",),
    "quartus_sh": ("--version",),
    "quartus_sta": ("--version",),
    "quartus_syn": ("--version",),
    "verilator": ("--version",),
    "vivado": ("-version",),
    "yosys": ("-V",),
}
LOCAL_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^\s(),]+)")
VERBOSITY_ENVIRONMENT = "LOOM_VERBOSE_LEVEL"


class CommandCacheError(RuntimeError):
    pass


def _verbosity() -> int:
    try:
        return max(0, int(os.environ.get(VERBOSITY_ENVIRONMENT, "0")))
    except ValueError:
        return 0


def _diagnostic(level: int, event: str, detail: str = "") -> None:
    if _verbosity() < level:
        return
    suffix = f" {detail}" if detail else ""
    print(f"[loom.hardware-tool-cache] {event}{suffix}", file=sys.stderr)


def _append_framed(digest: hashlib._Hash, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _domain_digest(domain: str, canonical: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode())
    digest.update(b"\0")
    _append_framed(digest, canonical)
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _tree_records(root: Path) -> Iterator[dict[str, object]]:
    def visit(directory: Path, relative: Path) -> Iterator[dict[str, object]]:
        try:
            members = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise CommandCacheError(f"cannot enumerate {directory}: {error}") from error
        for member in members:
            child_relative = relative / member.name
            child = Path(member.path)
            try:
                status = member.stat(follow_symlinks=False)
            except OSError as error:
                raise CommandCacheError(f"cannot inspect {child}: {error}") from error
            mode = stat.S_IMODE(status.st_mode)
            if stat.S_ISDIR(status.st_mode):
                yield {
                    "kind": "directory",
                    "mode": mode,
                    "path": child_relative.as_posix(),
                }
                yield from visit(child, child_relative)
            elif stat.S_ISREG(status.st_mode):
                yield {
                    "content_sha256": _file_digest(child),
                    "kind": "file",
                    "mode": mode,
                    "path": child_relative.as_posix(),
                    "size": status.st_size,
                }
            elif stat.S_ISLNK(status.st_mode):
                raise CommandCacheError(
                    f"tree contains a non-cacheable symbolic link: {child_relative}"
                )
            else:
                raise CommandCacheError(
                    f"tree contains a non-cacheable entry: {child_relative}"
                )

    yield from visit(root, Path())


def _tree_manifest(root: Path) -> bytes:
    records = list(_tree_records(root))
    return (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _external_arguments(arguments: list[str], working_directory: Path) -> list[Path]:
    external: list[Path] = []
    seen: set[Path] = set()
    for argument in arguments:
        candidate = Path(argument)
        if not candidate.is_absolute():
            continue
        candidate = candidate.resolve(strict=False)
        if candidate == working_directory or working_directory in candidate.parents:
            continue
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            external.append(candidate)
    return external


def _external_argument_digest(path: Path) -> tuple[str, str]:
    if path.is_dir() and not path.is_symlink():
        return "tree", hashlib.sha256(_tree_manifest(path)).hexdigest()
    if path.is_file() and not path.is_symlink():
        canonical = json.dumps(
            {"content_sha256": _file_digest(path), "size": path.stat().st_size},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "file", hashlib.sha256(canonical).hexdigest()
    raise CommandCacheError("external argument is not an ordinary file or tree")


def _input_material(working_directory: Path, external_arguments: list[Path]) -> bytes:
    external: list[dict[str, object]] = []
    for index, path in enumerate(external_arguments):
        kind, digest = _external_argument_digest(path)
        external.append(
            {
                "index": index,
                "kind": kind,
                "manifest_sha256": digest,
            }
        )
    document = {
        "external_arguments": external,
        "working_tree": json.loads(_tree_manifest(working_directory)),
    }
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _configuration(
    tool: str,
    arguments: list[str],
    working_directory: Path,
    external_arguments: list[Path],
) -> bytes:
    external_tokens = {
        path: f"$EXTERNAL_ARGUMENT_{index}"
        for index, path in enumerate(external_arguments)
    }
    normalized_arguments: list[str] = []
    for argument in arguments:
        candidate = Path(argument)
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
            if resolved == working_directory:
                normalized_arguments.append("$WORK")
                continue
            if working_directory in resolved.parents:
                normalized_arguments.append(
                    "$WORK/" + resolved.relative_to(working_directory).as_posix()
                )
                continue
            if resolved in external_tokens:
                normalized_arguments.append(external_tokens[resolved])
                continue
        normalized_arguments.append(argument)
    environment = {
        name: os.environ[name]
        for name in CONFIGURATION_ENVIRONMENT
        if name in os.environ
    }
    document = {
        "arguments": normalized_arguments,
        "environment": environment,
        "tool": tool,
    }
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _external_arguments_unchanged(
    external_arguments: list[Path], expected: list[tuple[str, str]]
) -> bool:
    try:
        return [
            _external_argument_digest(path) for path in external_arguments
        ] == expected
    except (CommandCacheError, OSError):
        return False


def _normalized_version(output: bytes) -> str:
    text = output.decode("utf-8", errors="strict").strip()
    return LOCAL_PATH_PATTERN.sub("$LOCAL_PATH", text)


def _tool_version(tool: str, executable: Path, environment: dict[str, str]) -> bytes:
    completed = subprocess.run(
        [str(executable), *VERSION_ARGUMENTS[tool]],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    if completed.returncode != 0 or not completed.stdout:
        raise CommandCacheError(f"cannot establish the exact {tool} version")
    document = {
        "executable_sha256": _file_digest(executable),
        "tool": tool,
        "version": _normalized_version(completed.stdout),
        "version_arguments": list(VERSION_ARGUMENTS[tool]),
    }
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _real_environment() -> tuple[Path, dict[str, str]]:
    real_path = os.environ.get(REAL_PATH_ENVIRONMENT)
    if not real_path:
        raise CommandCacheError(f"{REAL_PATH_ENVIRONMENT} is not set")
    environment = os.environ.copy()
    environment["PATH"] = real_path
    tool = Path(sys.argv[0]).name
    executable = shutil.which(tool, path=real_path)
    if executable is None:
        raise CommandCacheError(f"cannot resolve the real {tool} executable")
    return Path(executable).resolve(), environment


def _validated_working_directory() -> Path:
    root_spelling = os.environ.get(WORK_ROOT_ENVIRONMENT)
    if not root_spelling:
        raise CommandCacheError(f"{WORK_ROOT_ENVIRONMENT} is not set")
    root = Path(root_spelling).resolve()
    working_directory = Path.cwd().resolve()
    if working_directory == root or root not in working_directory.parents:
        raise CommandCacheError("working directory is outside the hardware test root")
    if working_directory.is_symlink() or not working_directory.is_dir():
        raise CommandCacheError("working directory is not an ordinary directory")
    return working_directory


def _cache_root() -> Path | None:
    spelling = os.environ.get(EXTERNAL_TOOL_CACHE_ROOT_ENVIRONMENT)
    if not spelling:
        return None
    selected = Path(spelling)
    if not selected.is_absolute():
        raise CommandCacheError("external-tool cache root must be absolute")
    if selected.is_symlink():
        raise CommandCacheError("external-tool cache root cannot be a symbolic link")
    root = selected.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or not root.is_dir() or root.stat().st_uid != os.geteuid():
        raise CommandCacheError("external-tool cache root is not private")
    root.chmod(0o700)
    initialization_lock = root / ".loom-external-tool-result-cache.lock"
    lock_flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    lock_descriptor = os.open(initialization_lock, lock_flags, 0o600)
    with os.fdopen(lock_descriptor, "a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = root / EXTERNAL_TOOL_CACHE_MARKER
        if marker.is_symlink():
            raise CommandCacheError("external-tool cache marker is invalid")
        if not marker.exists():
            other = {member.name for member in root.iterdir()} - {
                initialization_lock.name
            }
            if other:
                raise CommandCacheError(
                    "unmarked external-tool cache root is not empty"
                )
            marker.write_text(EXTERNAL_TOOL_CACHE_MARKER_CONTENTS, encoding="utf-8")
            marker.chmod(0o600)
        if (
            marker.is_symlink()
            or marker.read_text(encoding="utf-8") != EXTERNAL_TOOL_CACHE_MARKER_CONTENTS
        ):
            raise CommandCacheError("external-tool cache marker is incompatible")
        members = {member.name for member in root.iterdir()}
        if not members <= EXTERNAL_TOOL_CACHE_MEMBERS:
            raise CommandCacheError("external-tool cache contains a foreign member")
        for name in (COMMAND_ENTRY_DIRECTORY, COMMAND_LOCK_DIRECTORY):
            directory = root / name
            directory.mkdir(exist_ok=True, mode=0o700)
            if (
                directory.is_symlink()
                or not directory.is_dir()
                or directory.stat().st_uid != os.geteuid()
            ):
                raise CommandCacheError("external-tool cache namespace is invalid")
            directory.chmod(0o700)
    return root


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    completed = subprocess.run(
        ["cp", "-a", "--reflink=always", "--", f"{source}/.", str(destination)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise CommandCacheError("filesystem cannot reflink the command result")


def _replace_tree_members(source: Path, destination: Path, backup: Path) -> None:
    backup.mkdir()
    moved_original: list[Path] = []
    try:
        for member in list(destination.iterdir()):
            saved = backup / member.name
            member.rename(saved)
            moved_original.append(saved)
    except OSError:
        for saved in reversed(moved_original):
            saved.rename(destination / saved.name)
        raise

    installed: list[Path] = []
    try:
        for member in list(source.iterdir()):
            target = destination / member.name
            member.rename(target)
            installed.append(target)
    except OSError:
        for target in reversed(installed):
            target.rename(source / target.name)
        for saved in moved_original:
            saved.rename(destination / saved.name)
        raise
    shutil.rmtree(backup)


def _restore_entry(payload: Path, working_directory: Path) -> None:
    staging = Path(
        tempfile.mkdtemp(
            prefix=".loom-command-cache-restore-", dir=working_directory.parent
        )
    )
    try:
        restored = staging / "payload"
        _copy_tree(payload, restored)
        if _tree_manifest(restored) != _tree_manifest(payload):
            raise CommandCacheError("restored command payload changed while copying")
        _replace_tree_members(restored, working_directory, staging / "backup")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_all(descriptor: int, contents: bytes) -> None:
    remaining = memoryview(contents)
    while remaining:
        written = os.write(descriptor, remaining)
        remaining = remaining[written:]


def _entry_metadata(
    key: tuple[str, str, str], tree_digest: str, stdout: bytes, stderr: bytes
) -> bytes:
    document = {
        "execution_configuration_sha256": key[1],
        "input_material_sha256": key[0],
        "schema": CACHE_ENTRY_SCHEMA,
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "tool_version_sha256": key[2],
        "tree_sha256": tree_digest,
        "version": CACHE_ENTRY_VERSION,
    }
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()


def _read_entry(entry: Path, key: tuple[str, str, str]) -> tuple[bytes, bytes] | None:
    try:
        if entry.is_symlink() or not entry.is_dir():
            return None
        metadata_path = entry / "entry.json"
        stdout_path = entry / "stdout.bin"
        stderr_path = entry / "stderr.bin"
        payload = entry / "payload"
        for path in (metadata_path, stdout_path, stderr_path):
            if path.is_symlink() or not path.is_file():
                return None
        if payload.is_symlink() or not payload.is_dir():
            return None
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_bytes()
        expected = json.loads(
            _entry_metadata(key, metadata["tree_sha256"], stdout, stderr)
        )
        if metadata != expected:
            return None
        if (
            hashlib.sha256(_tree_manifest(payload)).hexdigest()
            != metadata["tree_sha256"]
        ):
            return None
        return stdout, stderr
    except (CommandCacheError, KeyError, OSError, TypeError, ValueError):
        return None


def _discard_entry(entry: Path) -> None:
    if entry.is_symlink() or not entry.is_dir():
        entry.unlink(missing_ok=True)
        return
    shutil.rmtree(entry)


def _publish_entry(
    entry: Path,
    key: tuple[str, str, str],
    working_directory: Path,
    stdout: bytes,
    stderr: bytes,
) -> None:
    parent = entry.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=parent))
    try:
        payload = staging / "payload"
        _copy_tree(working_directory, payload)
        tree_digest = hashlib.sha256(_tree_manifest(payload)).hexdigest()
        (staging / "stdout.bin").write_bytes(stdout)
        (staging / "stderr.bin").write_bytes(stderr)
        (staging / "entry.json").write_bytes(
            _entry_metadata(key, tree_digest, stdout, stderr)
        )
        staging.rename(entry)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _run_uncached(
    executable: Path, arguments: list[str], environment: dict[str, str]
) -> int:
    return subprocess.run([str(executable), *arguments], env=environment).returncode


def run() -> int:
    tool = Path(sys.argv[0]).name
    if tool not in SUPPORTED_TOOLS:
        raise CommandCacheError(f"unsupported hardware tool proxy name: {tool}")
    arguments = sys.argv[1:]
    executable, environment = _real_environment()
    if tuple(arguments) == VERSION_ARGUMENTS[tool]:
        return _run_uncached(executable, arguments, environment)
    try:
        cache_root = _cache_root()
    except (CommandCacheError, OSError) as error:
        _diagnostic(1, "unavailable", str(error))
        return _run_uncached(executable, arguments, environment)
    if cache_root is None:
        _diagnostic(1, "disabled")
        return _run_uncached(executable, arguments, environment)

    try:
        working_directory = _validated_working_directory()
        external = _external_arguments(arguments, working_directory)
        external_before = [_external_argument_digest(path) for path in external]
        key = (
            _domain_digest(
                "loom.external_tool_cache.input.v1",
                _input_material(working_directory, external),
            ),
            _domain_digest(
                "loom.external_tool_cache.configuration.v1",
                _configuration(tool, arguments, working_directory, external),
            ),
            _domain_digest(
                "loom.external_tool_cache.tool.v1",
                _tool_version(tool, executable, environment),
            ),
        )
    except (CommandCacheError, OSError) as error:
        _diagnostic(1, "unavailable", str(error))
        return _run_uncached(executable, arguments, environment)
    key_spelling = "/".join(key)
    _diagnostic(2, "key", key_spelling)
    entry = cache_root / COMMAND_ENTRY_DIRECTORY / key[0] / key[1] / key[2]
    lock_path = cache_root / COMMAND_LOCK_DIRECTORY / (".".join(key) + ".lock")
    lock_flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    except OSError as error:
        _diagnostic(1, "unavailable", f"cannot open exact-key lock: {error}")
        return _run_uncached(executable, arguments, environment)
    with os.fdopen(lock_descriptor, "a+b") as lock:
        try:
            os.fchmod(lock.fileno(), 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError as error:
            _diagnostic(1, "unavailable", f"cannot acquire exact-key lock: {error}")
            return _run_uncached(executable, arguments, environment)
        if entry.exists() or entry.is_symlink():
            cached = _read_entry(entry, key)
            if cached is not None:
                try:
                    _restore_entry(entry / "payload", working_directory)
                except (CommandCacheError, OSError):
                    pass
                else:
                    _diagnostic(1, "hit", key_spelling)
                    _write_all(sys.stdout.fileno(), cached[0])
                    _write_all(sys.stderr.fileno(), cached[1])
                    return 0
            else:
                _diagnostic(1, "discard", key_spelling)
                try:
                    _discard_entry(entry)
                except OSError:
                    pass

        _diagnostic(1, "miss", key_spelling)

        with tempfile.TemporaryDirectory(
            prefix=".attempt-", dir=cache_root / COMMAND_LOCK_DIRECTORY
        ) as attempt:
            stdout_path = Path(attempt) / "stdout.bin"
            stderr_path = Path(attempt) / "stderr.bin"
            with (
                stdout_path.open("wb") as stdout_stream,
                stderr_path.open("wb") as stderr_stream,
            ):
                completed = subprocess.run(
                    [str(executable), *arguments],
                    check=False,
                    env=environment,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                )
            stdout = stdout_path.read_bytes()
            stderr = stderr_path.read_bytes()
        _write_all(sys.stdout.fileno(), stdout)
        _write_all(sys.stderr.fileno(), stderr)
        if completed.returncode != 0:
            return completed.returncode
        if not _external_arguments_unchanged(external, external_before):
            _diagnostic(1, "publish-unavailable", "external input changed")
            return 0
        try:
            tool_after = _domain_digest(
                "loom.external_tool_cache.tool.v1",
                _tool_version(tool, executable, environment),
            )
        except (CommandCacheError, OSError) as error:
            _diagnostic(1, "publish-unavailable", str(error))
            return 0
        if tool_after != key[2]:
            _diagnostic(1, "publish-unavailable", "tool identity changed")
            return 0
        try:
            _publish_entry(entry, key, working_directory, stdout, stderr)
        except (CommandCacheError, OSError) as error:
            _diagnostic(1, "publish-unavailable", str(error))
        else:
            _diagnostic(2, "published", key_spelling)
        return 0


def main() -> int:
    try:
        return run()
    except CommandCacheError as error:
        print(f"hardware_tool_cache_invalid: {error}", file=sys.stderr)
        return 126


if __name__ == "__main__":
    sys.exit(main())
