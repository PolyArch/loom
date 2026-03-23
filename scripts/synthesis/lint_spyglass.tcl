# lint_spyglass.tcl -- SpyGlass RTL lint script.
#
# Loads the design and runs standard lint rules via SpyGlass.
#
# Required variables (set before invocation via sg_shell -f):
#   DESIGN_NAME   -- top-level module name
#   RTL_DIR       -- directory containing RTL sources
#
# Optional variables:
#   OUTPUT_DIR    -- directory for SpyGlass reports (default: ./spyglass_out)
#   FILELIST      -- path to filelist (default: RTL_DIR/filelist.f)
#   WAIVER_FILE   -- path to waiver file (optional)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
if {![info exists DESIGN_NAME]} {
    puts "ERROR: DESIGN_NAME must be set"
    exit 1
}
if {![info exists RTL_DIR]} {
    puts "ERROR: RTL_DIR must be set"
    exit 1
}
if {![info exists OUTPUT_DIR]} {
    set OUTPUT_DIR "./spyglass_out"
}
if {![info exists FILELIST]} {
    set FILELIST "$RTL_DIR/filelist.f"
}

# ---------------------------------------------------------------------------
# Create project
# ---------------------------------------------------------------------------
new_project $DESIGN_NAME -projectwdir $OUTPUT_DIR -force

# ---------------------------------------------------------------------------
# Read design sources
# ---------------------------------------------------------------------------
if {[file exists $FILELIST]} {
    read_file -type sourcelist $FILELIST
} else {
    # Read all SV files in the RTL directory
    foreach sv_file [glob -directory $RTL_DIR -type f *.sv] {
        read_file -type verilog $sv_file
    }
}

# Set top module
set_option top $DESIGN_NAME

# ---------------------------------------------------------------------------
# Configure lint methodology
# ---------------------------------------------------------------------------
# Standard lint rules
current_methodology $SPYGLASS_HOME/GuideWare/latest/block/rtl_handoff

# ---------------------------------------------------------------------------
# Waivers
# ---------------------------------------------------------------------------
if {[info exists WAIVER_FILE] && [file exists $WAIVER_FILE]} {
    read_file -type waiver $WAIVER_FILE
    puts "INFO: Waivers loaded from $WAIVER_FILE"
}

# ---------------------------------------------------------------------------
# Run lint
# ---------------------------------------------------------------------------
puts "INFO: Running SpyGlass lint on design '$DESIGN_NAME'"

run_goal lint/lint_rtl -overwrite

# ---------------------------------------------------------------------------
# Generate reports
# ---------------------------------------------------------------------------
file mkdir $OUTPUT_DIR

# Summary report
write_report moresimple > "$OUTPUT_DIR/lint_summary.rpt"

# Detailed violations
write_report detail > "$OUTPUT_DIR/lint_detail.rpt"

# ---------------------------------------------------------------------------
# Check results
# ---------------------------------------------------------------------------
set violation_count [get_result_count -severity error]
set warning_count   [get_result_count -severity warning]

puts ""
puts "=== SpyGlass Lint Summary ==="
puts "  Design:    $DESIGN_NAME"
puts "  Errors:    $violation_count"
puts "  Warnings:  $warning_count"
puts "  Reports:   $OUTPUT_DIR"
puts "=============================="

close_project -force

if {$violation_count > 0} {
    puts "FAIL: $violation_count lint errors found"
    exit 1
}

puts "PASS: SpyGlass lint clean"
exit 0
