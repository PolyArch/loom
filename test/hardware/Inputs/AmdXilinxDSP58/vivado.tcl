read_verilog -sv amd_xilinx_scalar_integer_multiply_system.sv
synth_design -top loom_module -part xcvp1802-vsva5601-3HP-e-S
set loom_dsp58_cells [get_cells -hierarchical -filter {REF_NAME == DSP58}]
if {[llength $loom_dsp58_cells] != 1} {
  error {synthesis did not preserve the exact DSP58 primitive}
}
opt_design
place_design
route_design
write_checkpoint -force routed.dcp
puts "LOOM_DSP58_CELLS [llength $loom_dsp58_cells]"
