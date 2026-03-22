// fabric_temporal_sw.sv -- Top-level temporal (tag-routed) switch.
//
// A tag-based routing switch with a configurable slot table.  Each
// input token carries a tag; the switch looks up the tag in a CAM
// table of NUM_SLOTS entries.  The first valid slot whose tag matches
// determines the routing: a per-slot bitmap selects which outputs the
// input drives.  Per-output round-robin arbitration resolves
// contention.  Broadcast atomicity ensures an input is consumed only
// when ALL its target outputs have accepted.
//
// Config layout (word-serial, low-to-high across the slot table):
//   For each slot s in [0, NUM_SLOTS):
//     valid(1) | tag(TAG_WIDTH) | route_bits(ROUTE_BITS)
//   where ROUTE_BITS = popcount(CONNECTIVITY).
//
// Port convention: all ports are !fabric.tagged (carry both data
// and tag fields).

module fabric_temporal_sw
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_IN      = 2,
  parameter int unsigned NUM_OUT     = 2,
  parameter int unsigned DATA_WIDTH  = 32,
  parameter int unsigned TAG_WIDTH   = 4,
  parameter int unsigned NUM_SLOTS   = 4,
  // Packed connectivity bit array.  Bit [o * NUM_IN + i] = 1 when
  // input i can physically reach output o.  Output-major, input-minor.
  parameter logic [NUM_OUT*NUM_IN-1:0] CONNECTIVITY = '1
)(
  input  logic                        clk,
  input  logic                        rst_n,

  // --- Config port (word-serial) ---
  input  logic                        cfg_valid,
  input  logic [31:0]                 cfg_wdata,
  output logic                        cfg_ready,

  // --- Input ports (tagged) ---
  input  logic                        in_valid  [0:NUM_IN-1],
  output logic                        in_ready  [0:NUM_IN-1],
  input  logic [DATA_WIDTH-1:0]       in_data   [0:NUM_IN-1],
  input  logic [TAG_WIDTH-1:0]        in_tag    [0:NUM_IN-1],

  // --- Output ports (tagged) ---
  output logic                        out_valid [0:NUM_OUT-1],
  input  logic                        out_ready [0:NUM_OUT-1],
  output logic [DATA_WIDTH-1:0]       out_data  [0:NUM_OUT-1],
  output logic [TAG_WIDTH-1:0]        out_tag   [0:NUM_OUT-1]
);

  // ---------------------------------------------------------------
  // Derived parameters
  // ---------------------------------------------------------------
  // Number of route bits per slot = number of connected positions.
  function automatic int unsigned count_ones(
    input logic [NUM_OUT*NUM_IN-1:0] vec
  );
    int unsigned cnt;
    integer idx;
    cnt = 0;
    for (idx = 0; idx < NUM_OUT * NUM_IN; idx = idx + 1) begin : popcount_loop
      cnt = cnt + {31'd0, vec[idx]};
    end : popcount_loop
    return cnt;
  endfunction : count_ones

  localparam int unsigned ROUTE_BITS = count_ones(CONNECTIVITY);

  // Per-slot bit width: valid(1) + tag(TAG_WIDTH) + route_bits.
  localparam int unsigned SLOT_WIDTH  = 1 + TAG_WIDTH + ROUTE_BITS;
  localparam int unsigned TOTAL_BITS  = NUM_SLOTS * SLOT_WIDTH;
  localparam int unsigned NUM_CFG_WORDS = (TOTAL_BITS + 31) / 32;
  localparam int unsigned SLOT_IDX_W  = clog2_min1(NUM_SLOTS);

  // ---------------------------------------------------------------
  // Config storage: word-serial shift register
  // ---------------------------------------------------------------
  logic [NUM_CFG_WORDS*32-1:0] cfg_sr;
  logic [$clog2(NUM_CFG_WORDS+1)-1:0] cfg_cnt;

  assign cfg_ready = 1'b1;

  always_ff @(posedge clk) begin : cfg_load
    if (!rst_n) begin : cfg_reset
      cfg_sr  <= '0;
      cfg_cnt <= '0;
    end : cfg_reset
    else begin : cfg_update
      if (cfg_valid && cfg_ready) begin : cfg_shift
        cfg_sr[cfg_cnt*32 +: 32] <= cfg_wdata;
        cfg_cnt <= cfg_cnt + 1'b1;
      end : cfg_shift
    end : cfg_update
  end : cfg_load

  // ---------------------------------------------------------------
  // Unpack slot table from config shift register
  // ---------------------------------------------------------------
  logic                  slot_valid  [0:NUM_SLOTS-1];
  logic [TAG_WIDTH-1:0]  slot_tag    [0:NUM_SLOTS-1];
  logic [ROUTE_BITS-1:0] slot_routes [0:NUM_SLOTS-1];

  always_comb begin : unpack_slots
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_SLOTS; iter_var0 = iter_var0 + 1) begin : per_slot
      slot_valid[iter_var0]  = cfg_sr[iter_var0 * SLOT_WIDTH];
      slot_tag[iter_var0]    = cfg_sr[iter_var0 * SLOT_WIDTH + 1 +: TAG_WIDTH];
      slot_routes[iter_var0] = cfg_sr[iter_var0 * SLOT_WIDTH + 1 + TAG_WIDTH +: ROUTE_BITS];
    end : per_slot
  end : unpack_slots

  // ---------------------------------------------------------------
  // Tag-matching CAM
  // ---------------------------------------------------------------
  logic                       match_found  [0:NUM_IN-1];
  logic [SLOT_IDX_W-1:0]     match_slot   [0:NUM_IN-1];
  logic [ROUTE_BITS-1:0]     match_routes [0:NUM_IN-1];

  fabric_temporal_sw_slot_match #(
    .NUM_IN      (NUM_IN),
    .TAG_WIDTH   (TAG_WIDTH),
    .NUM_SLOTS   (NUM_SLOTS),
    .ROUTE_BITS  (ROUTE_BITS),
    .SLOT_IDX_W  (SLOT_IDX_W)
  ) u_slot_match (
    .slot_valid   (slot_valid),
    .slot_tag     (slot_tag),
    .slot_routes  (slot_routes),
    .in_valid     (in_valid),
    .in_tag       (in_tag),
    .match_found  (match_found),
    .match_slot   (match_slot),
    .match_routes (match_routes)
  );

  // ---------------------------------------------------------------
  // Per-output arbitration with broadcast tracking
  // ---------------------------------------------------------------
  fabric_temporal_sw_arbiter #(
    .NUM_IN       (NUM_IN),
    .NUM_OUT      (NUM_OUT),
    .DATA_WIDTH   (DATA_WIDTH),
    .TAG_WIDTH    (TAG_WIDTH),
    .ROUTE_BITS   (ROUTE_BITS),
    .CONNECTIVITY (CONNECTIVITY)
  ) u_arbiter (
    .clk          (clk),
    .rst_n        (rst_n),
    .in_valid     (in_valid),
    .in_data      (in_data),
    .in_tag       (in_tag),
    .match_found  (match_found),
    .match_routes (match_routes),
    .out_valid    (out_valid),
    .out_ready    (out_ready),
    .out_data     (out_data),
    .out_tag      (out_tag),
    .in_ready     (in_ready)
  );

endmodule : fabric_temporal_sw
