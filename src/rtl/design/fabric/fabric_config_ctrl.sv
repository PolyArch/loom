// fabric_config_ctrl.sv -- Configuration word distribution controller.
//
// Receives a word-serial configuration stream from the host and routes
// each word to the appropriate hardware module based on a compile-time
// slice offset table.  The config stream is loaded during reset or
// explicit quiescent mode (not during active dataflow).
//
// The slice table is a parameter array: for each slice, it stores the
// starting word offset and word count.  Words are distributed in order
// to each module's config port.
//
// Config protocol:
//   - Host drives cfg_valid + cfg_wdata (32-bit words)
//   - Controller auto-increments an internal address counter
//   - Each module sees (mod_cfg_valid, mod_cfg_wdata, mod_cfg_word_idx)
//   - When all words delivered, controller asserts cfg_done

module fabric_config_ctrl #(
    parameter int unsigned NUM_SLICES     = 1,
    parameter int unsigned TOTAL_WORDS    = 1,
    // Slice table: SLICE_OFFSET[i] = starting word index for slice i
    // SLICE_COUNT[i] = number of words in slice i
    parameter int unsigned SLICE_OFFSET [NUM_SLICES] = '{default: 0},
    parameter int unsigned SLICE_COUNT  [NUM_SLICES] = '{default: 1}
)(
    input  logic        clk,
    input  logic        rst_n,

    // Host-side config input (word-serial)
    input  logic        cfg_valid,
    input  logic [31:0] cfg_wdata,
    output logic        cfg_ready,
    input  logic        cfg_last,

    // Per-slice config outputs
    output logic [NUM_SLICES-1:0]  slice_cfg_valid,
    output logic [31:0]            slice_cfg_wdata,
    output logic [15:0]            slice_cfg_word_idx,  // word index within slice

    // Status
    output logic        cfg_done
);

    // -------------------------------------------------------------------------
    // Address counter
    // -------------------------------------------------------------------------
    logic [31:0] word_counter;
    logic        loading;

    // Determine which slice the current word belongs to
    logic [NUM_SLICES-1:0] slice_match;
    logic [15:0]           word_within_slice;

    always_comb begin : slice_decode
        integer iter_var0;
        slice_match      = '0;
        word_within_slice = '0;
        for (iter_var0 = 0; iter_var0 < NUM_SLICES; iter_var0 = iter_var0 + 1) begin : decode_loop
            if (word_counter >= SLICE_OFFSET[iter_var0] &&
                word_counter < (SLICE_OFFSET[iter_var0] + SLICE_COUNT[iter_var0])) begin : match
                slice_match[iter_var0] = 1'b1;
                // Fabric width adaptation (WA-4): config bit extraction
                // See docs/spec-rtl-width-adaptation.md
                /* verilator lint_off WIDTHTRUNC */
                word_within_slice = word_counter[15:0] - SLICE_OFFSET[iter_var0][15:0];
                /* verilator lint_on WIDTHTRUNC */
            end : match
        end : decode_loop
    end : slice_decode

    // Config ready: accept words while loading
    assign cfg_ready = loading && !cfg_done;

    // Distribute to slices
    assign slice_cfg_valid    = (cfg_valid && cfg_ready) ? slice_match : '0;
    assign slice_cfg_wdata    = cfg_wdata;
    assign slice_cfg_word_idx = word_within_slice;

    always_ff @(posedge clk or negedge rst_n) begin : counter_proc
        if (!rst_n) begin : counter_reset
            word_counter <= '0;
            loading      <= 1'b1;
            cfg_done     <= 1'b0;
        end else begin : counter_active
            if (loading && !cfg_done) begin : counter_load
                if (cfg_valid && cfg_ready) begin : counter_advance
                    word_counter <= word_counter + 1;
                    if (cfg_last || (word_counter + 1 >= TOTAL_WORDS)) begin : counter_done
                        cfg_done <= 1'b1;
                        loading  <= 1'b0;
                    end : counter_done
                end : counter_advance
            end : counter_load
        end : counter_active
    end : counter_proc

endmodule : fabric_config_ctrl
