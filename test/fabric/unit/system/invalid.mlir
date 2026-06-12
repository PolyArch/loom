// RUN: loom %s -split-input-file -verify-diagnostics

fabric.system @bad_endpoint_soc memory_model = "sequential" {
  fabric.node @host0 kind = "host_core" ports = ["mem.aw:output"]
  // expected-error @+1 {{'fabric.link' op endpoint @missing does not refer to a fabric.node or fabric.external_port in this fabric.system}}
  fabric.link src = @host0 src_port = "mem" src_channel = "aw" dst = @missing dst_port = "mem" dst_channel = "aw"
}

// -----

fabric.system @bad_direction_soc memory_model = "sequential" {
  fabric.node @host0 kind = "host_core" ports = ["mem.aw:input"]
  fabric.node @dram0 kind = "memory" ports = ["host.aw:input"] attributes {bytes = 1024 : i64}
  // expected-error @+1 {{'fabric.link' op endpoint @host0 mem.aw is input, expected output}}
  fabric.link src = @host0 src_port = "mem" src_channel = "aw" dst = @dram0 dst_port = "host" dst_channel = "aw"
}

// -----

fabric.system @duplicate_source_soc memory_model = "sequential" {
  fabric.node @host0 kind = "host_core" ports = ["mem.aw:output"]
  fabric.node @dram0 kind = "memory" ports = ["host0.aw:input", "host1.aw:input"] attributes {bytes = 1024 : i64}
  fabric.link src = @host0 src_port = "mem" src_channel = "aw" dst = @dram0 dst_port = "host0" dst_channel = "aw"
  // expected-error @+1 {{'fabric.link' op endpoint @host0 mem.aw is used by more than one fabric.link}}
  fabric.link src = @host0 src_port = "mem" src_channel = "aw" dst = @dram0 dst_port = "host1" dst_channel = "aw"
}

// -----

fabric.system @bad_custom_memory_model_soc memory_model = "custom" {
  // expected-error @-1 {{'fabric.system' op memory_model 'custom' requires model_name or non-empty params}}
}

// -----

fabric.system @bad_spatial_reference_soc memory_model = "sequential" {
  // expected-error @+1 {{'fabric.node' op acc_core spatial reference @missing_template does not resolve to a fabric.module}}
  fabric.node @acc0 kind = "acc_core" ports = ["mem.aw:output"] attributes {spatial = @missing_template, scalar = "rv32im"}
}

// -----

fabric.system @bad_cache_params_soc memory_model = "sequential" {
  // expected-error @+1 {{'fabric.node' op kind 'cache' requires positive power-of-two line_bytes and capacity_bytes of at least one line}}
  fabric.node @l1d0 kind = "cache"
      ports = ["host.aw:input", "host.w:input", "host.b:output", "host.ar:input", "host.r:output",
               "mem.aw:output", "mem.w:output", "mem.b:input", "mem.ar:output", "mem.r:input"]
}

// -----

fabric.system @bad_cache_line_bytes_soc memory_model = "sequential" {
  // expected-error @+1 {{'fabric.node' op kind 'cache' requires positive power-of-two line_bytes and capacity_bytes of at least one line}}
  fabric.node @l1d0 kind = "cache"
      ports = ["host.aw:input", "host.w:input", "host.b:output", "host.ar:input", "host.r:output",
               "mem.aw:output", "mem.w:output", "mem.b:input", "mem.ar:output", "mem.r:input"]
      attributes {params = {line_bytes = 48 : i64, capacity_bytes = 32768 : i64}}
}

// -----

fabric.system @bad_cache_capacity_soc memory_model = "sequential" {
  // expected-error @+1 {{'fabric.node' op kind 'cache' requires positive power-of-two line_bytes and capacity_bytes of at least one line}}
  fabric.node @l1d0 kind = "cache"
      ports = ["host.aw:input", "host.w:input", "host.b:output", "host.ar:input", "host.r:output",
               "mem.aw:output", "mem.w:output", "mem.b:input", "mem.ar:output", "mem.r:input"]
      attributes {params = {line_bytes = 64 : i64, capacity_bytes = 1 : i64}}
}

// -----

fabric.system @bad_cache_ports_soc memory_model = "sequential" {
  // expected-error @+1 {{'fabric.node' op kind 'cache' requires at least one subordinate memory port and one manager memory port}}
  fabric.node @l1d0 kind = "cache"
      ports = ["host.aw:input", "host.w:input", "host.b:output", "host.ar:input", "host.r:output"]
      attributes {params = {line_bytes = 64 : i64, capacity_bytes = 32768 : i64}}
}
