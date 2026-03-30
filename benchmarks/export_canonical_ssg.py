#!/usr/bin/env python3
"""Materialize canonical S01..S18 SSG artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CanonicalArtifact:
    ssg_id: str
    source: str
    kind: str


DOMAIN_SOURCES = {
    "ai_llm": "benchmarks/tapestry/ai_llm/e01_formats/tdg_mlir.mlir",
    "arvr_stereo": "benchmarks/tapestry/arvr_stereo/e01_formats/tdg_mlir.mlir",
    "zk_stark": "benchmarks/tapestry/zk_stark/e01_formats/tdg_mlir.mlir",
    "dsp_ofdm": "benchmarks/tapestry/dsp_ofdm/e01_formats/tdg_mlir.mlir",
    "graph_analytics": "benchmarks/tapestry/graph_analytics/e01_formats/tdg_mlir.mlir",
}

CANONICAL_ARTIFACTS = [
    CanonicalArtifact("S01", DOMAIN_SOURCES["ai_llm"], "copy"),
    CanonicalArtifact("S02", DOMAIN_SOURCES["ai_llm"], "copy"),
    CanonicalArtifact("S03", DOMAIN_SOURCES["ai_llm"], "copy"),
    CanonicalArtifact("S04", DOMAIN_SOURCES["arvr_stereo"], "copy"),
    CanonicalArtifact("S05", DOMAIN_SOURCES["arvr_stereo"], "copy"),
    CanonicalArtifact("S06", DOMAIN_SOURCES["arvr_stereo"], "copy"),
    CanonicalArtifact("S07", DOMAIN_SOURCES["zk_stark"], "copy"),
    CanonicalArtifact("S08", DOMAIN_SOURCES["zk_stark"], "copy"),
    CanonicalArtifact("S09", DOMAIN_SOURCES["zk_stark"], "copy"),
    CanonicalArtifact("S10", DOMAIN_SOURCES["dsp_ofdm"], "copy"),
    CanonicalArtifact("S11", DOMAIN_SOURCES["dsp_ofdm"], "copy"),
    CanonicalArtifact("S12", DOMAIN_SOURCES["dsp_ofdm"], "copy"),
    CanonicalArtifact("S13", DOMAIN_SOURCES["graph_analytics"], "copy"),
    CanonicalArtifact("S14", DOMAIN_SOURCES["graph_analytics"], "copy"),
    CanonicalArtifact("S15", DOMAIN_SOURCES["graph_analytics"], "copy"),
    CanonicalArtifact("S16", "build/bin/jacobi_taskgraph", "binary"),
    CanonicalArtifact("S17", "build/bin/cg_taskgraph", "binary"),
    CanonicalArtifact("S18", "build/bin/nbody_taskgraph", "binary"),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main_repo_root() -> Path:
    root = repo_root()
    dot_git = root / ".git"
    if not dot_git.exists() or dot_git.is_dir():
        return root
    content = dot_git.read_text().strip()
    if not content.startswith("gitdir: "):
        return root
    gitdir = Path(content.split("gitdir: ", 1)[1].strip())
    if not gitdir.is_absolute():
        gitdir = (root / gitdir).resolve()
    worktree_dir = gitdir.parent
    git_dir = worktree_dir.parent
    return git_dir.parent


def build_root() -> Path:
    local_build = repo_root() / "build"
    if local_build.exists():
        return local_build
    return main_repo_root() / "build"


def build_bin_root() -> Path:
    return build_root() / "bin"


def write_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def write_binary_output(binary: Path, dst: Path) -> None:
    if not binary.exists():
        raise FileNotFoundError(f"missing taskgraph binary: {binary}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+b", dir=str(dst.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [str(binary), "-o", str(tmp_path)],
            cwd=str(repo_root()),
            check=True,
            capture_output=True,
            text=True,
        )
        tmp_path.replace(dst)
    finally:
        if tmp_path.exists() and tmp_path != dst:
            tmp_path.unlink(missing_ok=True)


def materialize(output_root: Path) -> None:
    root = repo_root()
    for artifact in CANONICAL_ARTIFACTS:
        dst = output_root / artifact.ssg_id / "ssg.mlir"
        src = root / artifact.source
        if artifact.kind == "copy":
            if not src.exists():
                raise FileNotFoundError(f"missing source artifact: {src}")
            write_copy(src, dst)
        elif artifact.kind == "binary":
            write_binary_output(build_bin_root() / Path(artifact.source).name, dst)
        else:
            raise ValueError(f"unknown artifact kind: {artifact.kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize canonical S01..S18 SSG artifacts")
    parser.add_argument(
        "--output-root",
        default="benchmarks",
        help="Directory that will receive S01..S18 subdirectories",
    )
    args = parser.parse_args()
    materialize(Path(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
