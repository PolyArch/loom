// Floating-point sine: result = sin(a)
module math_sin #(parameter int WIDTH = 32) (
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
      shortreal sa;
      real rr;
      always_comb begin : op
        sa = $bitstoshortreal(a_data);
        rr = $sin(real'(sa));
        result_data = $shortrealtobits(shortreal'(rr));
      end
    end else if (WIDTH == 64) begin : g_f64
      real ra, rr;
      always_comb begin : op
        ra = $bitstoreal(a_data);
        rr = $sin(ra);
        result_data = $realtobits(rr);
      end
    end else begin : g_unsupported
      initial $fatal(1, "math_sin: unsupported WIDTH=%0d", WIDTH);
      assign result_data = '0;
    end
  endgenerate
endmodule
