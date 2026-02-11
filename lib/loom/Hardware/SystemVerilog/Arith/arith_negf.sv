// Floating-point negation: result = -a
// Uses shortreal for 32-bit, real for 64-bit. Not synthesizable.
module arith_negf #(parameter int WIDTH = 32) (
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [WIDTH-1:0] a_data,
    output logic             result_valid,
    input  logic             result_ready,
    output logic [WIDTH-1:0] result_data
);
  assign result_valid = a_valid;
  assign a_ready = result_ready;

  generate
    if (WIDTH == 32) begin : g_f32
      shortreal sa, sr;
      always_comb begin : neg
        sa = $bitstoshortreal(a_data);
        sr = -sa;
        result_data = $shortrealtobits(sr);
      end
    end else if (WIDTH == 64) begin : g_f64
      real ra, rr;
      always_comb begin : neg
        ra = $bitstoreal(a_data);
        rr = -ra;
        result_data = $realtobits(rr);
      end
    end else begin : g_unsupported
      initial $fatal(1, "arith_negf: unsupported WIDTH=%0d (must be 32 or 64)", WIDTH);
      assign result_data = '0;
    end
  endgenerate
endmodule
