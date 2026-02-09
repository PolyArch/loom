//===-- fabric_map_tag.sv - Tag mapping module ----------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Translates the tag on a tagged stream via a CAM-style lookup table.
// Each table entry: {valid(1), src_tag(IN_TAG_WIDTH), dst_tag(OUT_TAG_WIDTH)}.
// On match, replaces the input tag with the destination tag.
//
// Errors:
//   CFG_MAP_TAG_DUP_TAG  - Duplicate valid source tags in the table
//   RT_MAP_TAG_NO_MATCH  - No valid entry matches the incoming tag
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_map_tag #(
    parameter int DATA_WIDTH    = 32,
    parameter int IN_TAG_WIDTH  = 4,
    parameter int OUT_TAG_WIDTH = 2,
    parameter int TABLE_SIZE    = 4,
    localparam int ENTRY_WIDTH  = 1 + IN_TAG_WIDTH + OUT_TAG_WIDTH,
    localparam int IN_PW        = (DATA_WIDTH + IN_TAG_WIDTH > 0) ? DATA_WIDTH + IN_TAG_WIDTH : 1,
    localparam int OUT_PW       = (DATA_WIDTH + OUT_TAG_WIDTH > 0) ? DATA_WIDTH + OUT_TAG_WIDTH : 1,
    localparam int CONFIG_WIDTH = TABLE_SIZE * ENTRY_WIDTH
) (
    input  logic                clk,
    input  logic                rst_n,

    // Streaming input (tagged with IN_TAG_WIDTH)
    input  logic                in_valid,
    output logic                in_ready,
    input  logic [IN_PW-1:0]   in_data,

    // Streaming output (tagged with OUT_TAG_WIDTH)
    output logic                out_valid,
    input  logic                out_ready,
    output logic [OUT_PW-1:0]   out_data,

    // Configuration: packed table entries
    input  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data,

    // Error output
    output logic                error_valid,
    output logic [15:0]         error_code
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation (COMP_ errors)
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (IN_TAG_WIDTH < 1)
      $fatal(1, "COMP_MAP_TAG_IN_TAG_WIDTH: IN_TAG_WIDTH must be >= 1");
    if (OUT_TAG_WIDTH < 1)
      $fatal(1, "COMP_MAP_TAG_OUT_TAG_WIDTH: OUT_TAG_WIDTH must be >= 1");
    if (TABLE_SIZE < 1)
      $fatal(1, "COMP_MAP_TAG_TABLE_SIZE: TABLE_SIZE must be >= 1");
    if (DATA_WIDTH < 1)
      $fatal(1, "COMP_MAP_TAG_DATA_WIDTH: DATA_WIDTH must be >= 1");
  end

  // -----------------------------------------------------------------------
  // Unpack table entries from cfg_data
  // -----------------------------------------------------------------------
  logic [TABLE_SIZE-1:0]                   entry_valid;
  logic [TABLE_SIZE-1:0][IN_TAG_WIDTH-1:0] entry_src_tag;
  logic [TABLE_SIZE-1:0][OUT_TAG_WIDTH-1:0] entry_dst_tag;

  always_comb begin : unpack_table
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < TABLE_SIZE; iter_var0 = iter_var0 + 1) begin : unpack_entry
      entry_valid[iter_var0]   = cfg_data[iter_var0 * ENTRY_WIDTH + OUT_TAG_WIDTH + IN_TAG_WIDTH];
      entry_src_tag[iter_var0] = cfg_data[iter_var0 * ENTRY_WIDTH + OUT_TAG_WIDTH +: IN_TAG_WIDTH];
      entry_dst_tag[iter_var0] = cfg_data[iter_var0 * ENTRY_WIDTH +: OUT_TAG_WIDTH];
    end
  end

  // -----------------------------------------------------------------------
  // Extract input tag and value
  // -----------------------------------------------------------------------
  logic [IN_TAG_WIDTH-1:0]  in_tag;
  logic [DATA_WIDTH-1:0]    in_value;

  assign in_tag   = in_data[DATA_WIDTH +: IN_TAG_WIDTH];
  assign in_value = in_data[DATA_WIDTH-1:0];

  // -----------------------------------------------------------------------
  // CAM lookup: find matching entry
  // -----------------------------------------------------------------------
  logic [TABLE_SIZE-1:0] match_vec;
  logic                  match_found;
  logic [OUT_TAG_WIDTH-1:0] matched_dst_tag;

  always_comb begin : cam_lookup
    integer iter_var0;
    match_vec = '0;
    for (iter_var0 = 0; iter_var0 < TABLE_SIZE; iter_var0 = iter_var0 + 1) begin : check_entry
      match_vec[iter_var0] = entry_valid[iter_var0] && (entry_src_tag[iter_var0] == in_tag);
    end
  end

  assign match_found = |match_vec;

  // Priority encoder: select first matching dst_tag
  always_comb begin : priority_select
    integer iter_var0;
    matched_dst_tag = '0;
    for (iter_var0 = 0; iter_var0 < TABLE_SIZE; iter_var0 = iter_var0 + 1) begin : select_entry
      if (match_vec[iter_var0]) begin : found
        matched_dst_tag = entry_dst_tag[iter_var0];
      end
    end
  end

  // -----------------------------------------------------------------------
  // Output assembly
  // -----------------------------------------------------------------------
  assign out_valid = in_valid && match_found;
  assign in_ready  = out_ready;
  assign out_data  = {matched_dst_tag, in_value};

  // -----------------------------------------------------------------------
  // Error detection
  // -----------------------------------------------------------------------

  // CFG error: duplicate valid source tags
  logic cfg_dup_tag;
  always_comb begin : dup_check
    integer iter_var0, iter_var1;
    cfg_dup_tag = 1'b0;
    for (iter_var0 = 0; iter_var0 < TABLE_SIZE; iter_var0 = iter_var0 + 1) begin : outer
      for (iter_var1 = iter_var0 + 1; iter_var1 < TABLE_SIZE; iter_var1 = iter_var1 + 1) begin : inner
        if (entry_valid[iter_var0] && entry_valid[iter_var1] &&
            (entry_src_tag[iter_var0] == entry_src_tag[iter_var1])) begin : dup_found
          cfg_dup_tag = 1'b1;
        end
      end
    end
  end

  // RT error: no match for valid input
  logic rt_no_match;
  assign rt_no_match = in_valid && !match_found;

  // Error latch: captures first error, held until reset
  always_ff @(posedge clk or negedge rst_n) begin : error_latch
    if (!rst_n) begin : reset
      error_valid <= 1'b0;
      error_code  <= 16'd0;
    end else if (!error_valid) begin : capture
      if (cfg_dup_tag) begin : dup_err
        error_valid <= 1'b1;
        error_code  <= CFG_MAP_TAG_DUP_TAG;
      end else if (rt_no_match) begin : match_err
        error_valid <= 1'b1;
        error_code  <= RT_MAP_TAG_NO_MATCH;
      end
    end
  end

endmodule
