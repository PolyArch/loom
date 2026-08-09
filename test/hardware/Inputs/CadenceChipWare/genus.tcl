foreach variable {
  LOOM_CHIPWARE_RTL
  LOOM_CHIPWARE_SYN_MODEL
  LOOM_CHIPWARE_LIBERTY
  LOOM_CHIPWARE_TOP
} {
  if {![info exists ::env($variable)]} {
    error "missing ChipWare Genus input: $variable"
  }
}

set_db library $::env(LOOM_CHIPWARE_LIBERTY)
read_hdl -sv [list \
  $::env(LOOM_CHIPWARE_SYN_MODEL) \
  $::env(LOOM_CHIPWARE_RTL)]
elaborate $::env(LOOM_CHIPWARE_TOP)
syn_generic
syn_map

set mappedInstances [get_db insts]
if {[llength $mappedInstances] == 0} {
  error "mapped ChipWare design has no instances"
}

write_hdl > mapped.v
report area > area.rpt
puts "LOOM_CHIPWARE_GENUS_PASS instances=[llength $mappedInstances]"
exit
