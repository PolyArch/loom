// fabric_temporal_sw_slot_match.sv -- Tag-matching CAM for temporal switch.
//
// For each input: parallel compare the input tag against all slot tags.
// Tag-as-matching semantics: scan slots in ascending order, the first
// valid slot whose tag matches wins.
//
// Outputs per input: matched slot index, match_found flag, and the
// route bitmap from the matched slot.

module fabric_temporal_sw_slot_match
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_IN      = 2,
  parameter int unsigned TAG_WIDTH   = 4,
  parameter int unsigned NUM_SLOTS   = 4,
  parameter int unsigned ROUTE_BITS  = 4,  // popcount(connectivity)
  // Derived: slot index width (at least 1 bit).  Exposed as parameter
  // so the port list can reference it.  Should not be overridden.
  parameter int unsigned SLOT_IDX_W = (NUM_SLOTS > 1) ? $clog2(NUM_SLOTS) : 1
)(
  // --- Slot table (unpacked from config) ---
  input  logic                       slot_valid   [0:NUM_SLOTS-1],
  input  logic [TAG_WIDTH-1:0]       slot_tag     [0:NUM_SLOTS-1],
  input  logic [ROUTE_BITS-1:0]      slot_routes  [0:NUM_SLOTS-1],

  // --- Per-input tag and valid ---
  input  logic                       in_valid     [0:NUM_IN-1],
  input  logic [TAG_WIDTH-1:0]       in_tag       [0:NUM_IN-1],

  // --- Match results per input ---
  output logic                       match_found  [0:NUM_IN-1],
  output logic [SLOT_IDX_W-1:0]      match_slot   [0:NUM_IN-1],
  output logic [ROUTE_BITS-1:0]      match_routes [0:NUM_IN-1]
);

  // ---------------------------------------------------------------
  // Combinational parallel CAM match
  // ---------------------------------------------------------------
  // For each input, scan all slots.  The first valid slot with a
  // matching tag wins (reverse iteration for priority-encoder idiom:
  // the lowest index that matches overwrites higher ones).

  always_comb begin : cam_match
    integer iter_var0;
    integer iter_var1;

    // Default: no match for any input.
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : default_no_match
      match_found[iter_var0]  = 1'b0;
      match_slot[iter_var0]   = '0;
      match_routes[iter_var0] = '0;
    end : default_no_match

    // Priority scan per input: iterate slots in reverse so that the
    // lowest-index match wins (it overwrites later assignments).
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : per_input
      if (in_valid[iter_var0]) begin : input_active
        for (iter_var1 = NUM_SLOTS - 1; iter_var1 >= 0; iter_var1 = iter_var1 - 1) begin : slot_scan
          if (slot_valid[iter_var1] && (slot_tag[iter_var1] == in_tag[iter_var0])) begin : slot_hit
            match_found[iter_var0]  = 1'b1;
            match_slot[iter_var0]   = iter_var1[SLOT_IDX_W-1:0];
            match_routes[iter_var0] = slot_routes[iter_var1];
          end : slot_hit
        end : slot_scan
      end : input_active
    end : per_input
  end : cam_match

endmodule : fabric_temporal_sw_slot_match
