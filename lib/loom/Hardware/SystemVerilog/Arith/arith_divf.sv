// Floating-point division: result = a / b
module arith_divf #(parameter int WIDTH = 32) (
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [WIDTH-1:0] a_data,
    input  logic             b_valid,
    output logic             b_ready,
    input  logic [WIDTH-1:0] b_data,
    output logic             result_valid,
    input  logic             result_ready,
    output logic [WIDTH-1:0] result_data
);
  generate
    if (WIDTH == 32) begin : g_f32
      shortreal sa, sb, sr;
      always_comb begin : div
        sa = $bitstoshortreal(a_data);
        sb = $bitstoshortreal(b_data);
        sr = sa / sb;
        result_data = $shortrealtobits(sr);
      end
    end else if (WIDTH == 64) begin : g_f64
      real ra, rb, rr;
      always_comb begin : div
        ra = $bitstoreal(a_data);
        rb = $bitstoreal(b_data);
        rr = ra / rb;
        result_data = $realtobits(rr);
      end
    end else begin : g_unsupported
      initial $fatal(1, "arith_divf: unsupported WIDTH=%0d", WIDTH);
      assign result_data = '0;
    end
  endgenerate
  assign result_valid = a_valid & b_valid;
  assign a_ready      = result_ready & b_valid;
  assign b_ready      = result_ready & a_valid;
endmodule
