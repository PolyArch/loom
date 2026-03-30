from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "benchmarks" / "export_canonical_ssg.py"


def load_export_module():
  spec = importlib.util.spec_from_file_location("export_canonical_ssg", SCRIPT_PATH)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to load {SCRIPT_PATH}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def run_export(output_root: Path) -> None:
  subprocess.run(
      [sys.executable, str(SCRIPT_PATH), "--output-root", str(output_root)],
      cwd=str(REPO_ROOT),
      check=True,
  )


def test_canonical_ssg_materialization(tmp_path):
  module = load_export_module()
  output_root = tmp_path / "benchmarks"
  run_export(output_root)

  for artifact in module.CANONICAL_ARTIFACTS:
    out_file = output_root / artifact.ssg_id / "ssg.mlir"
    assert out_file.exists(), f"missing canonical artifact {artifact.ssg_id}"
    assert out_file.stat().st_size > 0, f"empty canonical artifact {artifact.ssg_id}"

  for artifact in module.CANONICAL_ARTIFACTS[:15]:
    out_file = output_root / artifact.ssg_id / "ssg.mlir"
    src_file = REPO_ROOT / artifact.source
    assert out_file.read_bytes() == src_file.read_bytes(), (
        f"{artifact.ssg_id} should copy {artifact.source}"
    )

  for artifact in module.CANONICAL_ARTIFACTS[15:]:
    out_file = output_root / artifact.ssg_id / "ssg.mlir"
    expected = tmp_path / f"{artifact.ssg_id}.expected.mlir"
    bin_root = module.build_bin_root()
    subprocess.run(
        [str(bin_root / Path(artifact.source).name), "-o", str(expected)],
        cwd=str(REPO_ROOT),
        check=True,
    )
    assert out_file.read_bytes() == expected.read_bytes(), (
        f"{artifact.ssg_id} should match {artifact.source}"
    )
