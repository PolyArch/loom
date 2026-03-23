# dc_constraints.tcl -- Timing constraints for Synopsys DC synthesis.
#
# Required variables (set before sourcing):
#   CLOCK_PERIOD   -- clock period in ns (e.g., 10.0)
#   DESIGN_NAME    -- top-level module name
#
# Optional variables:
#   CLOCK_NAME        -- clock signal name (default: clk)
#   RESET_NAME        -- reset signal name (default: rst_n)
#   INPUT_DELAY_FRAC  -- input delay as fraction of clock period (default: 0.1)
#   OUTPUT_DELAY_FRAC -- output delay as fraction of clock period (default: 0.1)
#   NOC_PIPELINE_STAGES -- number of NoC router pipeline stages (default: 2)

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
if {![info exists CLOCK_PERIOD]} {
    set CLOCK_PERIOD 10.0
}
if {![info exists CLOCK_NAME]} {
    set CLOCK_NAME "clk"
}
if {![info exists RESET_NAME]} {
    set RESET_NAME "rst_n"
}
if {![info exists INPUT_DELAY_FRAC]} {
    set INPUT_DELAY_FRAC 0.1
}
if {![info exists OUTPUT_DELAY_FRAC]} {
    set OUTPUT_DELAY_FRAC 0.1
}
if {![info exists NOC_PIPELINE_STAGES]} {
    set NOC_PIPELINE_STAGES 2
}

# ---------------------------------------------------------------------------
# Clock definition
# ---------------------------------------------------------------------------
set clk_port [get_ports $CLOCK_NAME]
if {[sizeof_collection $clk_port] > 0} {
    create_clock -name $CLOCK_NAME -period $CLOCK_PERIOD $clk_port
    set_clock_uncertainty 0.1 [get_clocks $CLOCK_NAME]
    set_clock_transition  0.05 [get_clocks $CLOCK_NAME]
    puts "INFO: Clock '$CLOCK_NAME' period=${CLOCK_PERIOD}ns"
} else {
    puts "WARN: Clock port '$CLOCK_NAME' not found; creating virtual clock"
    create_clock -name $CLOCK_NAME -period $CLOCK_PERIOD
}

# ---------------------------------------------------------------------------
# Input / output delays
# ---------------------------------------------------------------------------
set input_delay  [expr {$CLOCK_PERIOD * $INPUT_DELAY_FRAC}]
set output_delay [expr {$CLOCK_PERIOD * $OUTPUT_DELAY_FRAC}]

# All inputs except clock
set all_in [remove_from_collection [all_inputs] [get_ports $CLOCK_NAME]]
if {[sizeof_collection $all_in] > 0} {
    set_input_delay $input_delay -clock $CLOCK_NAME $all_in
}

# All outputs
if {[sizeof_collection [all_outputs]] > 0} {
    set_output_delay $output_delay -clock $CLOCK_NAME [all_outputs]
}

# ---------------------------------------------------------------------------
# Reset false path
# ---------------------------------------------------------------------------
set rst_port [get_ports $RESET_NAME -quiet]
if {[sizeof_collection $rst_port] > 0} {
    set_false_path -from $rst_port
    puts "INFO: False path set on reset '$RESET_NAME'"
}

# ---------------------------------------------------------------------------
# NoC multicycle paths
# ---------------------------------------------------------------------------
# NoC router pipeline stages introduce known latency.
# Data paths through the NoC take multiple cycles by design.
set noc_router_cells [get_cells -hierarchical -filter "ref_name =~ *noc_router*" -quiet]
if {[sizeof_collection $noc_router_cells] > 0} {
    # Multicycle path for NoC router datapath (setup check)
    set_multicycle_path $NOC_PIPELINE_STAGES \
        -setup \
        -through $noc_router_cells

    # Multicycle path for NoC router datapath (hold check)
    set_multicycle_path [expr {$NOC_PIPELINE_STAGES - 1}] \
        -hold \
        -through $noc_router_cells

    puts "INFO: NoC multicycle paths set (${NOC_PIPELINE_STAGES} stages)"
}

# ---------------------------------------------------------------------------
# Configuration memory paths
# ---------------------------------------------------------------------------
# Configuration memory is loaded once at startup and is static during compute.
# Mark config_ctrl paths as false paths to avoid over-constraining.
set cfg_cells [get_cells -hierarchical -filter "ref_name =~ *config_ctrl*" -quiet]
if {[sizeof_collection $cfg_cells] > 0} {
    set_false_path -from $cfg_cells
    puts "INFO: False paths set on config_ctrl cells"
}

# ---------------------------------------------------------------------------
# Area constraint
# ---------------------------------------------------------------------------
set_max_area 0

# ---------------------------------------------------------------------------
# Design rule constraints
# ---------------------------------------------------------------------------
# Prevent high-fanout nets from degrading timing
set_max_fanout 20 [current_design]
set_max_transition [expr {$CLOCK_PERIOD * 0.1}] [current_design]

puts "INFO: dc_constraints.tcl complete"
