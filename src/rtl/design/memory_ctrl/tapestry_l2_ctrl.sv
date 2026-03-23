// tapestry_l2_ctrl.sv -- L2 controller with bank arbitration.
//
// Receives memory requests from multiple cores, decodes the bank
// address using interleaved addressing, queues requests per bank,
// performs round-robin arbitration across cores per bank, and routes
// responses back to the requesting core.

module tapestry_l2_ctrl
  import mem_ctrl_pkg::*;
#(
  parameter int unsigned NUM_BANKS        = 4,
  parameter int unsigned BANK_SIZE_BYTES  = 65536,
  parameter int unsigned DATA_WIDTH       = 32,
  parameter int unsigned INTERLEAVE_BYTES = 64,
  parameter int unsigned NUM_CORES        = 4,
  parameter int unsigned BANK_ADDR_WIDTH  = 16,
  parameter int unsigned ACCESS_LATENCY   = 4
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- Requests from cores (via NoC) ---
  input  mem_req_t    core_req       [NUM_CORES],
  input  logic        core_req_valid [NUM_CORES],
  output logic        core_req_ready [NUM_CORES],

  // --- Responses to cores (via NoC) ---
  output mem_resp_t   core_resp       [NUM_CORES],
  output logic        core_resp_valid [NUM_CORES],
  input  logic        core_resp_ready [NUM_CORES]
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned BANK_IDX_W = $clog2(NUM_BANKS > 1 ? NUM_BANKS : 2);
  localparam int unsigned CORE_IDX_W = $clog2(NUM_CORES > 1 ? NUM_CORES : 2);

  // ---------------------------------------------------------------
  // Per-bank request / response signals
  // ---------------------------------------------------------------
  mem_req_t   bank_req       [NUM_BANKS];
  logic       bank_req_valid [NUM_BANKS];
  logic       bank_req_ready [NUM_BANKS];

  mem_resp_t  bank_resp       [NUM_BANKS];
  logic       bank_resp_valid [NUM_BANKS];
  logic       bank_resp_ready [NUM_BANKS];

  // ---------------------------------------------------------------
  // Bank instantiation
  // ---------------------------------------------------------------
  generate
    genvar gb;
    for (gb = 0; gb < NUM_BANKS; gb = gb + 1) begin : gen_bank
      tapestry_l2_bank #(
        .BANK_SIZE_BYTES (BANK_SIZE_BYTES),
        .DATA_WIDTH      (DATA_WIDTH),
        .ADDR_WIDTH      (BANK_ADDR_WIDTH),
        .ACCESS_LATENCY  (ACCESS_LATENCY)
      ) u_bank (
        .clk        (clk),
        .rst_n      (rst_n),
        .req        (bank_req[gb]),
        .req_valid  (bank_req_valid[gb]),
        .req_ready  (bank_req_ready[gb]),
        .resp       (bank_resp[gb]),
        .resp_valid (bank_resp_valid[gb]),
        .resp_ready (bank_resp_ready[gb])
      );
    end : gen_bank
  endgenerate

  // ---------------------------------------------------------------
  // Address decoding: compute target bank for each core request
  // bank_id = (addr / INTERLEAVE_BYTES) % NUM_BANKS
  // ---------------------------------------------------------------
  logic [BANK_IDX_W-1:0] core_target_bank [NUM_CORES];

  generate
    genvar gc;
    for (gc = 0; gc < NUM_CORES; gc = gc + 1) begin : gen_bank_decode
      // Compute the interleave block index then take modulo NUM_BANKS.
      // For power-of-2 INTERLEAVE_BYTES and NUM_BANKS this becomes
      // a simple bit extraction.
      assign core_target_bank[gc] =
        core_req[gc].addr[$clog2(INTERLEAVE_BYTES) +: BANK_IDX_W];
    end : gen_bank_decode
  endgenerate

  // ---------------------------------------------------------------
  // Per-bank arbitration using round-robin arbiter
  //
  // For each bank, collect the request-valid signals from all cores
  // that target that bank, then arbitrate.
  // ---------------------------------------------------------------
  logic [NUM_CORES-1:0]  bank_core_req_vec   [NUM_BANKS];
  logic [NUM_CORES-1:0]  bank_core_grant_vec [NUM_BANKS];
  logic                  bank_arb_valid      [NUM_BANKS];
  logic [CORE_IDX_W-1:0] bank_arb_idx       [NUM_BANKS];

  // Build per-bank request vectors
  always_comb begin : build_bank_req_vec
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_BANKS; iter_var0 = iter_var0 + 1) begin : bank_loop
      for (iter_var1 = 0; iter_var1 < NUM_CORES; iter_var1 = iter_var1 + 1) begin : core_loop
        bank_core_req_vec[iter_var0][iter_var1] =
          core_req_valid[iter_var1] &
          (core_target_bank[iter_var1] == BANK_IDX_W'(iter_var0));
      end : core_loop
    end : bank_loop
  end : build_bank_req_vec

  // Per-bank round-robin arbiters
  /* verilator lint_off UNUSEDSIGNAL */
  generate
    genvar ga;
    for (ga = 0; ga < NUM_BANKS; ga = ga + 1) begin : gen_arb
      logic arb_ack;
      // Acknowledge when the bank accepts the request
      assign arb_ack = bank_arb_valid[ga] & bank_req_ready[ga];

      fabric_rr_arbiter #(
        .NUM_REQ (NUM_CORES)
      ) u_arb (
        .clk        (clk),
        .rst_n      (rst_n),
        .req        (bank_core_req_vec[ga]),
        .ack        (arb_ack),
        .grant      (bank_core_grant_vec[ga]),
        .grant_valid(bank_arb_valid[ga]),
        .grant_idx  (bank_arb_idx[ga])
      );
    end : gen_arb
  endgenerate
  /* verilator lint_on UNUSEDSIGNAL */

  // ---------------------------------------------------------------
  // Mux: route winning core request to bank
  // ---------------------------------------------------------------
  always_comb begin : bank_req_mux
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_BANKS; iter_var0 = iter_var0 + 1) begin : bank_mux_loop
      bank_req_valid[iter_var0] = bank_arb_valid[iter_var0];
      bank_req[iter_var0]       = core_req[bank_arb_idx[iter_var0]];
    end : bank_mux_loop
  end : bank_req_mux

  // ---------------------------------------------------------------
  // Core req_ready: a core is ready when its targeted bank grants it
  // ---------------------------------------------------------------
  always_comb begin : core_ready_gen
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_CORES; iter_var0 = iter_var0 + 1) begin : core_ready_loop
      core_req_ready[iter_var0] = 1'b0;
      for (iter_var1 = 0; iter_var1 < NUM_BANKS; iter_var1 = iter_var1 + 1) begin : bank_check_loop
        if (bank_core_grant_vec[iter_var1][iter_var0] &&
            bank_req_ready[iter_var1]) begin : grant_ready
          core_req_ready[iter_var0] = 1'b1;
        end : grant_ready
      end : bank_check_loop
    end : core_ready_loop
  end : core_ready_gen

  // ---------------------------------------------------------------
  // Response routing: route bank responses to the requesting core
  //
  // Each bank response carries core_id in the resp struct. We route
  // each bank response to the matching core output port.
  // ---------------------------------------------------------------
  always_comb begin : resp_route
    integer iter_var0;
    integer iter_var1;

    // Default: no valid responses
    for (iter_var0 = 0; iter_var0 < NUM_CORES; iter_var0 = iter_var0 + 1) begin : resp_default_loop
      core_resp_valid[iter_var0] = 1'b0;
      core_resp[iter_var0]       = '0;
    end : resp_default_loop

    // Default: accept all bank responses (overridden below if core
    // cannot accept)
    for (iter_var0 = 0; iter_var0 < NUM_BANKS; iter_var0 = iter_var0 + 1) begin : bank_resp_default
      bank_resp_ready[iter_var0] = 1'b1;
    end : bank_resp_default

    // Route each bank response to the target core
    for (iter_var0 = 0; iter_var0 < NUM_BANKS; iter_var0 = iter_var0 + 1) begin : resp_bank_loop
      if (bank_resp_valid[iter_var0]) begin : resp_bank_check
        // The core_id field identifies the destination core
        for (iter_var1 = 0; iter_var1 < NUM_CORES; iter_var1 = iter_var1 + 1) begin : resp_core_match
          if (bank_resp[iter_var0].core_id == 4'(iter_var1)) begin : resp_match
            core_resp_valid[iter_var1] = 1'b1;
            core_resp[iter_var1]       = bank_resp[iter_var0];
            bank_resp_ready[iter_var0] = core_resp_ready[iter_var1];
          end : resp_match
        end : resp_core_match
      end : resp_bank_check
    end : resp_bank_loop
  end : resp_route

endmodule : tapestry_l2_ctrl
