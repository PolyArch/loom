foreach variable {
  LOOM_CHIPWARE_MAPPED_NETLIST
  LOOM_CHIPWARE_TOP
  LOOM_CHIPWARE_TECH_LEF
  LOOM_CHIPWARE_CELL_LEF
} {
  if {![info exists ::env($variable)]} {
    error "missing ChipWare Innovus input: $variable"
  }
}

read_physical -lef [list \
  $::env(LOOM_CHIPWARE_TECH_LEF) \
  $::env(LOOM_CHIPWARE_CELL_LEF)]
read_netlist $::env(LOOM_CHIPWARE_MAPPED_NETLIST) \
  -top $::env(LOOM_CHIPWARE_TOP)
init_design

set mappedInstances [get_db insts]
if {[llength $mappedInstances] == 0} {
  error "mapped ChipWare design has no physical instances"
}

place_design
write_db innovus_db
write_def placed.def
puts "LOOM_CHIPWARE_INNOVUS_PASS instances=[llength $mappedInstances]"
exit
