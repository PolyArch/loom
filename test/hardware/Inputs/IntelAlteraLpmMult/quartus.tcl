set project loom_intel_altera_lpm_mult
project_new $project -revision $project -overwrite
set_global_assignment -name FAMILY "Agilex 7"
set_global_assignment -name DEVICE AGIA040R39A1E1VC
set_global_assignment -name TOP_LEVEL_ENTITY loom_module
set_global_assignment -name SYSTEMVERILOG_FILE intel_altera_scalar_integer_multiply_system.sv
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
foreach port {
  clock reset
  input_0_data[*] input_0_valid input_1_data[*] input_1_valid
  output_0_ready input_0_ready input_1_ready output_0_data[*] output_0_valid
  configuration_0[*]
} {
  set_instance_assignment -name VIRTUAL_PIN ON -to $port
}
project_close
