# dc_compile.tcl -- Main Synopsys DC synthesis script.
#
# Reads RTL sources, applies constraints, runs compile_ultra, and
# generates area/timing/power reports plus gate-level netlist.
#
# Required variables (set before invocation via dc_shell -f):
#   DESIGN_NAME   -- top-level module name
#   SV_FILES      -- Tcl list of SystemVerilog source file paths
#   OUTPUT_DIR    -- directory for reports and output files
#
# Optional variables:
#   PDK_TARGET    -- PDK selection: saed14, asap7, saed32 (default: saed14)
#   CLOCK_PERIOD  -- clock period in ns (default: 10.0)
#   CLOCK_NAME    -- clock port name (default: clk)
#   RESET_NAME    -- reset port name (default: rst_n)
#   COMPILE_EFFORT -- compile effort: high, medium (default: high)
#   HIERARCHICAL  -- if 1, use -no_autoungroup (default: 0)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
if {![info exists DESIGN_NAME]} {
    puts "ERROR: DESIGN_NAME must be set"
    exit 1
}
if {![info exists SV_FILES]} {
    puts "ERROR: SV_FILES must be set"
    exit 1
}
if {![info exists OUTPUT_DIR]} {
    puts "ERROR: OUTPUT_DIR must be set"
    exit 1
}
if {![info exists PDK_TARGET]} {
    set PDK_TARGET "saed14"
}
if {![info exists CLOCK_PERIOD]} {
    set CLOCK_PERIOD 10.0
}
if {![info exists CLOCK_NAME]} {
    set CLOCK_NAME "clk"
}
if {![info exists RESET_NAME]} {
    set RESET_NAME "rst_n"
}
if {![info exists COMPILE_EFFORT]} {
    set COMPILE_EFFORT "high"
}
if {![info exists HIERARCHICAL]} {
    set HIERARCHICAL 0
}

# ---------------------------------------------------------------------------
# Create output directory
# ---------------------------------------------------------------------------
file mkdir $OUTPUT_DIR

# ---------------------------------------------------------------------------
# Library setup
# ---------------------------------------------------------------------------
set script_dir [file dirname [file normalize [info script]]]
source "$script_dir/dc_setup.tcl"

# ---------------------------------------------------------------------------
# Read RTL sources
# ---------------------------------------------------------------------------
puts "INFO: Reading [llength $SV_FILES] source files for design '$DESIGN_NAME'"

foreach sv_file $SV_FILES {
    if {![file exists $sv_file]} {
        puts "WARN: Source file not found: $sv_file"
        continue
    }
    puts "  analyze: $sv_file"
    analyze -format sverilog $sv_file
}

# ---------------------------------------------------------------------------
# Elaborate and link
# ---------------------------------------------------------------------------
elaborate $DESIGN_NAME
current_design $DESIGN_NAME
link

if {[link] == 0} {
    puts "ERROR: Design link failed"
    exit 1
}

puts "INFO: Design '$DESIGN_NAME' elaborated and linked"

# ---------------------------------------------------------------------------
# Apply timing constraints
# ---------------------------------------------------------------------------
source "$script_dir/dc_constraints.tcl"

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
puts "INFO: Starting compile_ultra (effort=$COMPILE_EFFORT, hierarchical=$HIERARCHICAL)"

if {$HIERARCHICAL} {
    compile_ultra -no_autoungroup
} else {
    compile_ultra
}

puts "INFO: Compile complete"

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
puts "INFO: Generating reports in $OUTPUT_DIR"

report_area -hierarchy > "$OUTPUT_DIR/area_report.rpt"
report_timing -max_paths 20 -sort_by slack > "$OUTPUT_DIR/timing_report.rpt"
report_timing -delay_type min -max_paths 10 > "$OUTPUT_DIR/timing_hold_report.rpt"
report_power -analysis_effort high > "$OUTPUT_DIR/power_report.rpt"
report_constraint -all_violators > "$OUTPUT_DIR/constraint_report.rpt"
report_qor > "$OUTPUT_DIR/qor_report.rpt"
report_reference -hierarchy > "$OUTPUT_DIR/reference_report.rpt"

# Cell count summary
report_cell > "$OUTPUT_DIR/cell_report.rpt"

# ---------------------------------------------------------------------------
# Write netlist and design database
# ---------------------------------------------------------------------------
change_names -rules verilog -hierarchy
write -format verilog -hierarchy -output "$OUTPUT_DIR/${DESIGN_NAME}_netlist.v"
write -format ddc -hierarchy -output "$OUTPUT_DIR/${DESIGN_NAME}.ddc"

# Write SDC for downstream tools (PrimeTime, ICC2)
write_sdc "$OUTPUT_DIR/${DESIGN_NAME}.sdc"

puts "INFO: Netlist written to $OUTPUT_DIR/${DESIGN_NAME}_netlist.v"
puts "INFO: DDC written to $OUTPUT_DIR/${DESIGN_NAME}.ddc"
puts "INFO: SDC written to $OUTPUT_DIR/${DESIGN_NAME}.sdc"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
set slack [get_attribute [get_timing_paths -max_paths 1] slack]
puts ""
puts "=== Synthesis Summary ==="
puts "  Design:       $DESIGN_NAME"
puts "  PDK:          $PDK_TARGET"
puts "  Clock period: ${CLOCK_PERIOD} ns"
puts "  Worst slack:  $slack"
puts "  Output dir:   $OUTPUT_DIR"
puts "========================="

exit
