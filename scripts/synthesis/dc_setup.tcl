# dc_setup.tcl -- Library setup for Synopsys DC synthesis.
#
# Supports three PDK targets selected via the PDK_TARGET variable:
#   saed14  -- SAED14nm FinFET (primary)
#   asap7   -- ASAP7 7nm (secondary)
#   saed32  -- SAED32 32nm (reference baseline)
#
# Required variables (set before sourcing):
#   PDK_TARGET   -- one of: saed14, asap7, saed32
#
# Optional variables:
#   SAED14_ROOT  -- override for SAED14nm EDK path
#   ASAP7_ROOT   -- override for ASAP7 path
#   SAED32_ROOT  -- override for SAED32 EDK path

# ---------------------------------------------------------------------------
# Default library root paths (can be overridden by caller)
# ---------------------------------------------------------------------------

if {![info exists SAED14_ROOT]} {
    set SAED14_ROOT "/mnt/nas0/eda.libs/saed14/EDK_08_2025"
}
if {![info exists ASAP7_ROOT]} {
    set ASAP7_ROOT "/mnt/nas0/eda.libs/asap7"
}
if {![info exists SAED32_ROOT]} {
    set SAED32_ROOT "/mnt/nas0/eda.libs/saed32/EDK_08_2025"
}

if {![info exists PDK_TARGET]} {
    set PDK_TARGET "saed14"
}

# ---------------------------------------------------------------------------
# SAED14nm FinFET 14nm -- Primary target
# ---------------------------------------------------------------------------
proc setup_saed14 {root} {
    set std_rvt_lib "$root/SAED14nm_EDK_STD_RVT/liberty/nldm/base"
    set std_hvt_lib "$root/SAED14nm_EDK_STD_HVT/liberty/nldm/base"
    set std_lvt_lib "$root/SAED14nm_EDK_STD_LVT/liberty/nldm/base"
    set sram_lib    "$root/SAED14nm_EDK_SRAM/liberty/nldm"

    # Target library: RVT typical-typical corner for synthesis
    set_app_var target_library [list \
        "$std_rvt_lib/saed14rvt_base_tt0p8v25c.db" \
    ]

    # Link library includes SRAM macros
    set_app_var link_library [list "*"]
    lappend link_library "$std_rvt_lib/saed14rvt_base_tt0p8v25c.db"

    # Add SRAM macro libraries if available
    if {[file exists "$sram_lib/saed14sram_tt0p8v25c.db"]} {
        lappend link_library "$sram_lib/saed14sram_tt0p8v25c.db"
    }

    # Search path for Verilog models
    set_app_var search_path [list \
        "$root/SAED14nm_EDK_STD_RVT/verilog" \
        "$root/SAED14nm_EDK_SRAM/verilog" \
    ]

    # Multi-Vt: additional libraries for optimization
    # DC can swap cells between Vt flavors during compile_ultra
    set multi_vt_libs {}
    if {[file exists "$std_hvt_lib/saed14hvt_base_tt0p8v25c.db"]} {
        lappend multi_vt_libs "$std_hvt_lib/saed14hvt_base_tt0p8v25c.db"
    }
    if {[file exists "$std_lvt_lib/saed14lvt_base_tt0p8v25c.db"]} {
        lappend multi_vt_libs "$std_lvt_lib/saed14lvt_base_tt0p8v25c.db"
    }
    foreach lib $multi_vt_libs {
        lappend link_library $lib
    }
    set_app_var link_library $link_library

    puts "INFO: SAED14nm library setup complete"
    puts "  target_library: $target_library"
    puts "  SRAM macros:    $sram_lib"
}

# ---------------------------------------------------------------------------
# ASAP7 7nm -- Secondary target
# ---------------------------------------------------------------------------
proc setup_asap7 {root} {
    # Leverage the existing setup7.tcl from the ASAP7 distribution
    set setup_script "$root/setup7.tcl"
    if {[file exists $setup_script]} {
        source $setup_script
        puts "INFO: ASAP7 library setup via setup7.tcl complete"
        return
    }

    # Fallback: manual setup
    set cell_path "$root/asap7sc7p5t_28"

    set_app_var target_library [list \
        "$cell_path/LIB/NLDM/asap7sc7p5t_SIMPLE_RVT_TT_nldm.db" \
        "$cell_path/LIB/NLDM/asap7sc7p5t_INVBUF_RVT_TT_nldm.db" \
        "$cell_path/LIB/NLDM/asap7sc7p5t_AO_RVT_TT_nldm.db" \
        "$cell_path/LIB/NLDM/asap7sc7p5t_SEQ_RVT_TT_nldm.db" \
    ]

    set sram_path "$root/ASAP7_SRAM_0p0/generated/LIB"
    set_app_var link_library [list "*"]
    foreach lib $target_library {
        lappend link_library $lib
    }
    # Add SRAM macros if available
    foreach db [glob -nocomplain "$sram_path/*.db"] {
        lappend link_library $db
    }
    set_app_var link_library $link_library

    set_app_var search_path "$cell_path/Verilog"

    puts "INFO: ASAP7 library setup (fallback) complete"
}

# ---------------------------------------------------------------------------
# SAED32 32nm -- Reference baseline
# ---------------------------------------------------------------------------
proc setup_saed32 {root} {
    set lib_path "$root/lib/stdcell_rvt/db_nldm"

    set_app_var target_library [list \
        "$lib_path/saed32rvt_ff0p85v25c.db" \
    ]

    set_app_var link_library [list "*" "$lib_path/saed32rvt_ff0p85v25c.db"]

    set_app_var search_path [list "$lib_path"]

    puts "INFO: SAED32 library setup complete"
}

# ---------------------------------------------------------------------------
# Dispatch based on PDK_TARGET
# ---------------------------------------------------------------------------
switch -exact $PDK_TARGET {
    "saed14" {
        setup_saed14 $SAED14_ROOT
    }
    "asap7" {
        setup_asap7 $ASAP7_ROOT
    }
    "saed32" {
        setup_saed32 $SAED32_ROOT
    }
    default {
        puts "ERROR: Unknown PDK_TARGET '$PDK_TARGET'. Use: saed14, asap7, saed32"
        exit 1
    }
}

# Enable multi-threading
set_host_options -max_cores 8

puts "INFO: dc_setup.tcl complete for PDK=$PDK_TARGET"
