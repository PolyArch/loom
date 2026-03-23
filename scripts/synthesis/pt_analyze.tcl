# pt_analyze.tcl -- PrimeTime timing and power analysis.
#
# Reads gate-level netlist from DC synthesis output and performs
# static timing analysis and power estimation.
#
# Required variables (set before invocation via pt_shell -f):
#   DESIGN_NAME   -- top-level module name
#   NETLIST_DIR   -- directory containing DC synthesis output
#
# Optional variables:
#   PDK_TARGET    -- PDK selection: saed14, asap7, saed32 (default: saed14)
#   OUTPUT_DIR    -- directory for PT reports (default: NETLIST_DIR/pt_reports)
#   VCD_FILE      -- VCD file for switching activity annotation
#   SAIF_FILE     -- SAIF file for switching activity annotation

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
if {![info exists DESIGN_NAME]} {
    puts "ERROR: DESIGN_NAME must be set"
    exit 1
}
if {![info exists NETLIST_DIR]} {
    puts "ERROR: NETLIST_DIR must be set"
    exit 1
}
if {![info exists PDK_TARGET]} {
    set PDK_TARGET "saed14"
}
if {![info exists OUTPUT_DIR]} {
    set OUTPUT_DIR "$NETLIST_DIR/pt_reports"
}
if {![info exists VCD_FILE]} {
    set VCD_FILE ""
}
if {![info exists SAIF_FILE]} {
    set SAIF_FILE ""
}

file mkdir $OUTPUT_DIR

# ---------------------------------------------------------------------------
# Library setup (reuse DC setup)
# ---------------------------------------------------------------------------
set script_dir [file dirname [file normalize [info script]]]
source "$script_dir/dc_setup.tcl"

# ---------------------------------------------------------------------------
# Read gate-level netlist
# ---------------------------------------------------------------------------
set netlist_file "$NETLIST_DIR/${DESIGN_NAME}_netlist.v"
set sdc_file "$NETLIST_DIR/${DESIGN_NAME}.sdc"

if {![file exists $netlist_file]} {
    puts "ERROR: Netlist not found: $netlist_file"
    exit 1
}

read_verilog $netlist_file
current_design $DESIGN_NAME
link_design

# ---------------------------------------------------------------------------
# Apply timing constraints
# ---------------------------------------------------------------------------
if {[file exists $sdc_file]} {
    read_sdc $sdc_file
    puts "INFO: SDC constraints loaded from $sdc_file"
} else {
    puts "WARN: No SDC file found, applying default constraints"
    source "$script_dir/dc_constraints.tcl"
}

# ---------------------------------------------------------------------------
# Static timing analysis
# ---------------------------------------------------------------------------
puts "INFO: Running static timing analysis"

# Setup timing
report_timing -max_paths 50 -sort_by slack \
    -path_type full_clock \
    > "$OUTPUT_DIR/timing_setup.rpt"

# Hold timing
report_timing -delay_type min -max_paths 50 -sort_by slack \
    -path_type full_clock \
    > "$OUTPUT_DIR/timing_hold.rpt"

# Constraint violations
report_constraint -all_violators > "$OUTPUT_DIR/constraint_violations.rpt"

# Clock network analysis
report_clock_timing -type summary > "$OUTPUT_DIR/clock_summary.rpt"

puts "INFO: Timing reports written to $OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------
puts "INFO: Running power analysis"

# Annotate switching activity from VCD or SAIF if available
set activity_annotated 0

if {$SAIF_FILE ne "" && [file exists $SAIF_FILE]} {
    read_saif $SAIF_FILE -strip_path "testbench/DUT"
    set activity_annotated 1
    puts "INFO: SAIF activity annotated from $SAIF_FILE"
} elseif {$VCD_FILE ne "" && [file exists $VCD_FILE]} {
    read_vcd $VCD_FILE -strip_path "testbench/DUT"
    set activity_annotated 1
    puts "INFO: VCD activity annotated from $VCD_FILE"
}

if {!$activity_annotated} {
    # Use default switching activity estimation
    set_switching_activity -static_probability 0.5 -toggle_rate 0.1 [all_inputs]
    puts "INFO: Using estimated switching activity (no VCD/SAIF)"
}

# Power reports
report_power -hierarchy > "$OUTPUT_DIR/power_hierarchy.rpt"
report_power > "$OUTPUT_DIR/power_summary.rpt"

puts "INFO: Power reports written to $OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
puts ""
puts "=== PrimeTime Analysis Summary ==="
puts "  Design:          $DESIGN_NAME"
puts "  PDK:             $PDK_TARGET"
puts "  Activity source: [expr {$activity_annotated ? \"VCD/SAIF\" : \"estimated\"}]"
puts "  Reports:         $OUTPUT_DIR"
puts "==================================="

exit
