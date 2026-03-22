# synth_template.tcl -- Synopsys DC synthesis template for fabric RTL modules.
#
# Variables to be substituted by run_synth.py:
#   ${DESIGN_NAME}  -- top-level module name
#   ${SV_FILES}     -- space-separated list of .sv source files
#   ${OUTPUT_DIR}   -- output directory for reports and netlist
#
# Target library: saed32 (TSMC 32nm educational)

# Library setup
set search_path [list /mnt/nas0/eda.libs/saed32/EDK_08_2025/lib/stdcell_rvt/db_nldm]
set target_library {saed32rvt_ff0p85v25c.db}
set link_library [list * $target_library]

# Read design
set sv_file_list {${SV_FILES}}
foreach sv_file $sv_file_list {
    analyze -format sverilog $sv_file
}
elaborate ${DESIGN_NAME}

# Link design
link

# Basic timing constraints (relaxed -- proving synthesizability, not closure)
create_clock -name clk -period 10.0 [get_ports clk]
set_input_delay 1.0 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 1.0 -clock clk [all_outputs]

# Reset is async active-low
set_false_path -from [get_ports rst_n]

# Compile
compile_ultra -no_autoungroup

# Reports
report_area -hierarchy > ${OUTPUT_DIR}/area_report.rpt
report_timing -max_paths 10 > ${OUTPUT_DIR}/timing_report.rpt
report_power > ${OUTPUT_DIR}/power_report.rpt
report_constraint -all_violators > ${OUTPUT_DIR}/constraint_report.rpt

# Write netlist
write -format verilog -hierarchy -output ${OUTPUT_DIR}/synth_netlist.v
write -format ddc -hierarchy -output ${OUTPUT_DIR}/synth.ddc

exit
