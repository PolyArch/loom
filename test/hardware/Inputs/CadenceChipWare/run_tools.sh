#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 <generated-rtl> <simulation-model> <synthesis-model> <liberty> <technology-lef> <cell-lef> <work-directory>" >&2
  exit 2
fi

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
generated_rtl=$(realpath "$1")
simulation_model=$(realpath "$2")
synthesis_model=$(realpath "$3")
liberty=$(realpath "$4")
technology_lef=$(realpath "$5")
cell_lef=$(realpath "$6")
mkdir -p "$7"
work_directory=$(realpath "$7")

xrun_binary=${XRUN_BIN:-xrun}
genus_binary=${GENUS_BIN:-genus}
innovus_binary=${INNOVUS_BIN:-innovus}
top=chipware_scalar_integer_multiply

mkdir -p "$work_directory/xcelium" "$work_directory/genus" \
  "$work_directory/innovus"

(
  cd "$work_directory/xcelium"
  "$xrun_binary" -64bit -sv -clean \
    -top chipware_scalar_integer_multiply_testbench \
    "$simulation_model" "$generated_rtl" \
    "$script_directory/cw_mult_testbench.sv" > xrun.log 2>&1
)
grep -Eq '^LOOM_CHIPWARE_XCELIUM_PASS vectors=65536$' \
  "$work_directory/xcelium/xrun.log"

(
  cd "$work_directory/genus"
  env LOOM_CHIPWARE_RTL="$generated_rtl" \
    LOOM_CHIPWARE_SYN_MODEL="$synthesis_model" \
    LOOM_CHIPWARE_LIBERTY="$liberty" LOOM_CHIPWARE_TOP="$top" \
    "$genus_binary" -batch -files "$script_directory/genus.tcl" \
    > genus.stdout.log 2>&1
)
grep -Eq '^LOOM_CHIPWARE_GENUS_PASS instances=[1-9][0-9]*$' \
  "$work_directory/genus/genus.stdout.log"

(
  cd "$work_directory/innovus"
  env LOOM_CHIPWARE_MAPPED_NETLIST="$work_directory/genus/mapped.v" \
    LOOM_CHIPWARE_TOP="$top" LOOM_CHIPWARE_TECH_LEF="$technology_lef" \
    LOOM_CHIPWARE_CELL_LEF="$cell_lef" \
    "$innovus_binary" -stylus -no_gui -files \
    "$script_directory/innovus.tcl" > innovus.stdout.log 2>&1
)
grep -Eq '^LOOM_CHIPWARE_INNOVUS_PASS instances=[1-9][0-9]*$' \
  "$work_directory/innovus/innovus.stdout.log"

echo "LOOM_CHIPWARE_TOOL_CONSUMPTION_PASS"
