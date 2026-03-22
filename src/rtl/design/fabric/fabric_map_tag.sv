// fabric_map_tag.sv -- Tag lookup rewrite via configurable CAM table.
//
// Parallel CAM match on the input tag.  The first valid entry whose
// src_tag matches the input tag wins; its dst_tag replaces the tag.
// When no entry matches, the module blocks (out_valid=0, in_ready=0).
//
// Config layout per entry (low-to-high): valid(1), src_tag(IN_TAG_WIDTH),
// dst_tag(OUT_TAG_WIDTH).  Entries are loaded word-serially; the full
// table is bit-packed across 32-bit config words.

module fabric_map_tag
  import fabric_pkg::*;
#(
  parameter int unsigned DATA_WIDTH    = 32,
  parameter int unsigned TABLE_SIZE    = 4,
  parameter int unsigned IN_TAG_WIDTH  = 4,
  parameter int unsigned OUT_TAG_WIDTH = 4
)(
  input  logic                        clk,
  input  logic                        rst_n,

  // --- Config port (word-serial) ---
  input  logic                        cfg_valid,
  input  logic [31:0]                 cfg_wdata,
  output logic                        cfg_ready,

  // --- Input (tagged value, tag width = IN_TAG_WIDTH) ---
  input  logic                        in_valid,
  output logic                        in_ready,
  input  logic [DATA_WIDTH-1:0]       in_data,
  input  logic [IN_TAG_WIDTH-1:0]     in_tag,

  // --- Output (tagged value, tag width = OUT_TAG_WIDTH) ---
  output logic                        out_valid,
  input  logic                        out_ready,
  output logic [DATA_WIDTH-1:0]       out_data,
  output logic [OUT_TAG_WIDTH-1:0]    out_tag
);

  // ---------------------------------------------------------------
  // Config storage
  // ---------------------------------------------------------------
  // Per-entry bit width: valid(1) + src_tag + dst_tag.
  localparam int unsigned ENTRY_WIDTH    = 1 + IN_TAG_WIDTH + OUT_TAG_WIDTH;
  localparam int unsigned TOTAL_BITS     = TABLE_SIZE * ENTRY_WIDTH;
  // Number of 32-bit config words needed (word-aligned per spec).
  localparam int unsigned NUM_CFG_WORDS  = (TOTAL_BITS + 31) / 32;
  localparam int unsigned CFG_SR_WIDTH   = NUM_CFG_WORDS * 32;

  // Config storage as individual 32-bit word registers.
  // Each word is managed by a generate-block always_ff to avoid
  // multi-driver issues and to allow clean per-word reset.
  logic [31:0] cfg_words [0:NUM_CFG_WORDS-1];

  // Word index counter for sequential loading.
  localparam int unsigned CNT_W = clog2_min1(NUM_CFG_WORDS);
  logic [CNT_W-1:0] cfg_word_idx;

  assign cfg_ready = 1'b1;

  // Sequential counter for config word index.
  always_ff @(posedge clk) begin : cfg_cnt_seq
    if (!rst_n) begin : cnt_reset
      cfg_word_idx <= '0;
    end : cnt_reset
    else begin : cnt_update
      if (cfg_valid && cfg_ready) begin : cnt_advance
        // Fabric width adaptation (WA-4): config bit extraction
        // See docs/spec-rtl-width-adaptation.md
        /* verilator lint_off WIDTHTRUNC */
        if (cfg_word_idx == CNT_W'(NUM_CFG_WORDS - 1)) begin : cnt_wrap
        /* verilator lint_on WIDTHTRUNC */
          cfg_word_idx <= '0;
        end : cnt_wrap
        else begin : cnt_inc
          cfg_word_idx <= cfg_word_idx + 1'b1;
        end : cnt_inc
      end : cnt_advance
    end : cnt_update
  end : cfg_cnt_seq

  // Per-word config storage via generate.
  generate
    genvar gw;
    for (gw = 0; gw < NUM_CFG_WORDS; gw = gw + 1) begin : gen_cfg_word
      always_ff @(posedge clk) begin : cfg_word_seq
        if (!rst_n) begin : word_reset
          cfg_words[gw] <= 32'b0;
        end : word_reset
        else begin : word_update
          // Fabric width adaptation (WA-4): config bit extraction
          // See docs/spec-rtl-width-adaptation.md
          /* verilator lint_off WIDTHTRUNC */
          if (cfg_valid && cfg_ready && (cfg_word_idx == CNT_W'(gw))) begin : word_write
          /* verilator lint_on WIDTHTRUNC */
            cfg_words[gw] <= cfg_wdata;
          end : word_write
        end : word_update
      end : cfg_word_seq
    end : gen_cfg_word
  endgenerate

  // Flatten config words into a single bit vector for unpacking.
  logic [CFG_SR_WIDTH-1:0] cfg_flat;

  always_comb begin : flatten_cfg
    integer iter_var0;
    cfg_flat = '0;
    for (iter_var0 = 0; iter_var0 < NUM_CFG_WORDS; iter_var0 = iter_var0 + 1) begin : flat_word
      cfg_flat[iter_var0*32 +: 32] = cfg_words[iter_var0];
    end : flat_word
  end : flatten_cfg

  // ---------------------------------------------------------------
  // Unpack table entries from config storage
  // ---------------------------------------------------------------
  logic                       tbl_valid     [0:TABLE_SIZE-1];
  logic [IN_TAG_WIDTH-1:0]    tbl_src_tag   [0:TABLE_SIZE-1];
  logic [OUT_TAG_WIDTH-1:0]   tbl_dst_tag   [0:TABLE_SIZE-1];

  // CAM match results.
  logic                       cam_hit       [0:TABLE_SIZE-1];
  logic                       any_match;
  logic [OUT_TAG_WIDTH-1:0]   matched_dst_tag;

  always_comb begin : unpack_and_match
    integer iter_var0;
    any_match       = 1'b0;
    matched_dst_tag = '0;

    // Fabric width adaptation (WA-4): config bit extraction
    // See docs/spec-rtl-width-adaptation.md
    /* verilator lint_off WIDTHTRUNC */
    for (iter_var0 = 0; iter_var0 < TABLE_SIZE; iter_var0 = iter_var0 + 1) begin : entry_unpack
      tbl_valid[iter_var0]   = cfg_flat[iter_var0 * ENTRY_WIDTH];
      tbl_src_tag[iter_var0] = cfg_flat[iter_var0 * ENTRY_WIDTH + 1 +: IN_TAG_WIDTH];
      tbl_dst_tag[iter_var0] = cfg_flat[iter_var0 * ENTRY_WIDTH + 1 + IN_TAG_WIDTH +: OUT_TAG_WIDTH];

      cam_hit[iter_var0] = tbl_valid[iter_var0] &&
                           (tbl_src_tag[iter_var0] == in_tag);
    end : entry_unpack
    /* verilator lint_on WIDTHTRUNC */

    // Priority encoder: first valid match (lowest index) wins.
    // Reverse iteration so lower-index overwrites higher-index.
    for (iter_var0 = TABLE_SIZE - 1; iter_var0 >= 0; iter_var0 = iter_var0 - 1) begin : priority_sel
      if (cam_hit[iter_var0]) begin : match_found
        any_match       = 1'b1;
        matched_dst_tag = tbl_dst_tag[iter_var0];
      end : match_found
    end : priority_sel
  end : unpack_and_match

  // ---------------------------------------------------------------
  // Output logic: block when no match
  // ---------------------------------------------------------------
  assign out_valid = in_valid & any_match;
  assign in_ready  = out_ready & any_match;
  assign out_data  = in_data;
  assign out_tag   = matched_dst_tag;

endmodule : fabric_map_tag
