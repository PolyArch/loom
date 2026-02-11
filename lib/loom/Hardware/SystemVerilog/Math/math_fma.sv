// Floating-point fused multiply-add: result = a * b + c
module math_fma #(parameter int WIDTH = 32) (
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [WIDTH-1:0] a_data,
    input  logic             b_valid,
    output logic             b_ready,
    input  logic [WIDTH-1:0] b_data,
    input  logic             c_valid,
    output logic             c_ready,
    input  logic [WIDTH-1:0] c_data,
    output logic             result_valid,
    input  logic             result_ready,
    output logic [WIDTH-1:0] result_data
);
  generate
    if (WIDTH == 32) begin : g_f32
      shortreal sa, sb, sc, sr;
      always_comb begin : op
        sa = $bitstoshortreal(a_data);
        sb = $bitstoshortreal(b_data);
        sc = $bitstoshortreal(c_data);
        sr = sa * sb + sc;
        result_data = $shortrealtobits(sr);
      end
    end else if (WIDTH == 64) begin : g_f64
      real ra, rb, rc, rr;
      always_comb begin : op
        ra = $bitstoreal(a_data);
        rb = $bitstoreal(b_data);
        rc = $bitstoreal(c_data);
        rr = ra * rb + rc;
        result_data = $realtobits(rr);
      end
    end else begin : g_unsupported
      initial $fatal(1, "math_fma: unsupported WIDTH=%0d", WIDTH);
      assign result_data = '0;
    end
  endgenerate
  assign result_valid = a_valid & b_valid & c_valid;
  assign a_ready      = result_ready & b_valid & c_valid;
  assign b_ready      = result_ready & a_valid & c_valid;
  assign c_ready      = result_ready & a_valid & b_valid;
endmodule
