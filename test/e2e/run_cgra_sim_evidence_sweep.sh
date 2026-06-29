#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'USAGE'
usage: run_cgra_sim_evidence_sweep.sh --output-dir DIR [--case NAME]... [--hardware-source checked-in|dotproduct-fmuladd|byte-swap-store|shared-vector-alu|shared-vector-math|shared-memory-reduction|shared-signal-window|adg-builder] [--legacy-app-root DIR] [--jobs N]
USAGE
}

OUT_DIR=""
HARDWARE_SOURCE="checked-in"
LEGACY_APP_ROOT="${ROOT}/temp/old_implementation_loom/loom/tests/app"
JOBS_ARG=""
declare -a CASES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUT_DIR="${2:?missing --output-dir value}"
      shift 2
      ;;
    --case)
      CASES+=("${2:?missing --case value}")
      shift 2
      ;;
    --hardware-source)
      HARDWARE_SOURCE="${2:?missing --hardware-source value}"
      shift 2
      ;;
    --legacy-app-root)
      LEGACY_APP_ROOT="${2:?missing --legacy-app-root value}"
      shift 2
      ;;
    --jobs)
      JOBS_ARG="${2:?missing --jobs value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUT_DIR}" ]]; then
  echo "--output-dir is required" >&2
  usage >&2
  exit 2
fi

if [[ ${#CASES[@]} -eq 0 ]]; then
  CASES=(
    autocorrelation
    vecsum
    vecsum-while
    dotproduct
    dotprod
    dot_product_3d
    axpy
    binary_search
    bitonic_stage
    bitonic_stage-tweak
    bit_reverse
    bisection_step
    clz
    ctz
    downsample
    downsample_avg
    delta_encode
    delta_decode
    find_first_set
    prefix_sum
    cumsum
    prefix_sum_inclusive
    prefix_sum_exclusive
    pack_bits
    parity
    partition
    popcount
    unpack_bits
    integrate_trapz
    reduction
    mean
    vecnorm_l1
    vecnorm_l2
    correlation
    covariance
    compare_swap
    compact
    hash_mix
    string_hash
    merge
    modexp
    modmul
    spmv
    sort_bubble
    convolve_1d
    conv1d
    conv2d
    convolve_1d_same
    crc32
    cross_product
    quat_mult
    fir_filter
    fir_filter_stateful
    gather
    gf_mul
    gemv
    gemm
    matmul
    mmtile
    mat3x3_mult
    spmspv
    stream_update
    lower_bound
    matvec
    moving_avg
    newton_iter
    outer
    byte_swap
    scatter_add
    sort_insertion
    xor_block
    relu
    rotate_bits
    rle_decode
    rle_encode
    runge_kutta_step
    sbox_lookup
    softmax
    transpose
    transform_point
    upper_bound
    upsample
    vecadd
    vecmul
    vecscale
    variance
  )
fi

validate_unique_cases() {
  local case_name
  declare -A seen=()
  for case_name in "${CASES[@]}"; do
    if [[ -n "${seen[${case_name}]:-}" ]]; then
      echo "duplicate --case: ${case_name}" >&2
      exit 2
    fi
    seen["${case_name}"]=1
  done
}

validate_unique_cases

mkdir -p "${OUT_DIR}"
chain_root="${OUT_DIR}/_chains"
mkdir -p "${chain_root}"

default_jobs() {
  local value="${JOBS_ARG:-${LOOM_TEST_JOBS:-${JOBS:-}}}"
  if [[ -z "${value}" ]]; then
    value="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  fi
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "invalid --jobs value: ${value}" >&2
    exit 2
  fi
  printf '%s\n' "${value}"
}

PARALLEL_JOBS="$(default_jobs)"

normalize_case_artifacts() {
  local case_name="$1"
  local case_out="$2"
  python3 - "${case_name}" "${case_out}" "${OUT_DIR}" <<'PY'
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


case_name = sys.argv[1]
case_out = Path(sys.argv[2])
out_dir = Path(sys.argv[3])


def artifact_id(path: Path) -> str:
    name = path.name
    for suffix in (".csv", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def wait_for_file(path: Path) -> None:
    for _ in range(50):
        if path.is_file():
            return
        time.sleep(0.1)
    raise SystemExit(f"missing component artifact {path}")


def read_json(path: Path) -> dict:
    last_error: Exception | None = None
    for _ in range(50):
        if not path.is_file():
            time.sleep(0.1)
            continue
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            last_error = exc
            time.sleep(0.1)
    if last_error is not None:
        raise SystemExit(f"artifact is not valid JSON after retry: {path}: {last_error}")
    raise SystemExit(f"missing component artifact {path}")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_component(identity: str, id_map: dict[str, str]) -> None:
    if identity in id_map:
        return
    source = case_out / f"{identity}.json"
    wait_for_file(source)
    namespaced = f"{case_name}.{identity}"
    shutil.copy2(source, out_dir / f"{namespaced}.json")
    id_map[identity] = namespaced


def rewrite_components(data: dict, fields: tuple[str, ...]) -> dict:
    id_map: dict[str, str] = {}
    for field in fields:
        identities = data.get(field)
        if not isinstance(identities, list):
            continue
        for identity in identities:
            if isinstance(identity, str) and identity:
                copy_component(identity, id_map)
        data[field] = [id_map.get(identity, identity) for identity in identities]
    fingerprints = data.get("input_artifact_fingerprints")
    if isinstance(fingerprints, dict):
        rewritten = {}
        for identity, value in fingerprints.items():
            if isinstance(identity, str):
                rewritten[id_map.get(identity, identity)] = value
        data["input_artifact_fingerprints"] = rewritten
    return data


dfg = rewrite_components(
    read_json(case_out / f"{case_name}-dfg-sim-report.json"),
    ("component_dfg_sim_report_identities",),
)
mapping = rewrite_components(
    read_json(case_out / "pnr-mapping.json"),
    ("component_mapping_artifact_identities",),
)
cgra = rewrite_components(
    read_json(case_out / f"{case_name}-cgra-sim-report.json"),
    ("component_dfg_sim_report_identities", "component_cgra_sim_report_identities"),
)

dfg_dest = out_dir / f"{case_name}.dfg.report.json"
mapping_dest = out_dir / f"{case_name}.mapping.json"
cgra_dest = out_dir / f"{case_name}.cgra.report.json"
write_json(dfg_dest, dfg)
write_json(mapping_dest, mapping)
write_json(cgra_dest, cgra)

comparison = read_json(case_out / "sim-comparison-report.json")
comparison["dfg_sim_report_identity"] = artifact_id(dfg_dest)
comparison["mapping_artifact_identity"] = artifact_id(mapping_dest)
comparison["cgra_sim_report_identity"] = artifact_id(cgra_dest)
comparison["comparison_id"] = (
    f"sim-comparison::{case_name}::{comparison['cgra_sim_report_identity']}"
)
write_json(out_dir / f"{case_name}.sim-comparison-report.json", comparison)

for path in (dfg_dest, mapping_dest, cgra_dest):
    if not path.is_file() or not fingerprint(path):
        raise SystemExit(f"failed to emit {path}")
PY
}

case_aggregate_status() {
  local case_name="$1"
  python3 - "${case_name}" "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path


case_name = sys.argv[1]
out_dir = Path(sys.argv[2])


def status(suffix: str) -> str:
    path = out_dir / f"{case_name}.{suffix}"
    if not path.is_file():
        return "missing"
    data = json.loads(path.read_text())
    value = data.get("status")
    return value if isinstance(value, str) and value else "missing"


statuses = {
    "dfg": status("dfg.report.json"),
    "mapping": status("mapping.json"),
    "cgra": status("cgra.report.json"),
    "comparison": status("sim-comparison-report.json"),
}
if all(value == "pass" for value in statuses.values()):
    print("pass")
elif any(value == "fail" for value in statuses.values()):
    print("fail")
elif any(value == "unsupported" for value in statuses.values()):
    print("unsupported")
else:
    print("blocked")
PY
}

run_case_job() {
  local case_name="$1"
  local case_out="${chain_root}/${case_name}"
  rm -rf "${case_out}"
  local case_hardware_source="${HARDWARE_SOURCE}"
  if [[ "${HARDWARE_SOURCE}" == "checked-in" && "${case_name}" == "vecscale" ]]; then
    case_hardware_source="shared-vector-alu"
  fi
  mkdir -p "${case_out}"
  (
    if ! bash "${ROOT}/test/e2e/run_intermediate_artifact_chain.sh" \
        --output-dir "${case_out}" \
        --case "${case_name}" \
        --hardware-source "${case_hardware_source}" \
        --legacy-app-root "${LEGACY_APP_ROOT}" \
        > "${case_out}/chain.stdout.log" \
        2> "${case_out}/chain.stderr.log"; then
      echo "fail" > "${case_out}/job.status"
      exit 1
    fi
    if ! normalize_case_artifacts "${case_name}" "${case_out}" \
        > "${case_out}/normalize.stdout.log" \
        2> "${case_out}/normalize.stderr.log"; then
      echo "fail" > "${case_out}/job.status"
      exit 1
    fi
    echo "[${case_name}] $(case_aggregate_status "${case_name}")" > "${case_out}/status.line"
    echo "pass" > "${case_out}/job.status"
  ) &
}

print_case_failure() {
  local case_name="$1"
  local case_out="${chain_root}/${case_name}"
  echo "[${case_name}] failed" >&2
  for log in \
      "${case_out}/chain.stdout.log" \
      "${case_out}/chain.stderr.log" \
      "${case_out}/normalize.stdout.log" \
      "${case_out}/normalize.stderr.log"; do
    if [[ -s "${log}" ]]; then
      cat "${log}" >&2
    fi
  done
}

active_jobs=0
job_failed=0
for case_name in "${CASES[@]}"; do
  run_case_job "${case_name}"
  active_jobs=$((active_jobs + 1))
  if (( active_jobs >= PARALLEL_JOBS )); then
    if ! wait -n; then
      job_failed=1
    fi
    active_jobs=$((active_jobs - 1))
  fi
done
while (( active_jobs > 0 )); do
  if ! wait -n; then
    job_failed=1
  fi
  active_jobs=$((active_jobs - 1))
done

for case_name in "${CASES[@]}"; do
  case_out="${chain_root}/${case_name}"
  if [[ -f "${case_out}/status.line" ]]; then
    cat "${case_out}/status.line"
  else
    job_failed=1
    print_case_failure "${case_name}"
  fi
done

if (( job_failed != 0 )); then
  exit 1
fi
