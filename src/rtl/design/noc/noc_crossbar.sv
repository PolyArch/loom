// noc_crossbar.sv -- NUM_PORTS x NUM_PORTS crossbar for NoC router.
//
// Multiplexer-based crossbar.  Each output port has an independent
// select input (from the switch arbiter) that picks which input port
// to route.  When sel_valid is deasserted for an output, that output
// produces no valid flit.

module noc_crossbar
  import noc_pkg::*;
#(
  parameter int unsigned DATA_WIDTH = NOC_DATA_WIDTH_DEFAULT,
  parameter int unsigned NUM_PORTS  = NOC_NUM_PORTS
)(
  // Input flits from all input ports.
  input  logic [flit_width(DATA_WIDTH)-1:0]  in_flits  [NUM_PORTS],

  // Per-output-port selection.
  input  logic [$clog2(NUM_PORTS)-1:0]       sel       [NUM_PORTS],
  input  logic                               sel_valid [NUM_PORTS],

  // Output flits.
  output logic [flit_width(DATA_WIDTH)-1:0]  out_flits [NUM_PORTS],
  output logic                               out_valid [NUM_PORTS]
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned SEL_W  = $clog2(NUM_PORTS);

  // ---------------------------------------------------------------
  // Crossbar mux -- one mux per output port
  // ---------------------------------------------------------------
  always_comb begin : xbar_mux
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : out_port_loop
      out_flits[iter_var0] = '0;
      out_valid[iter_var0] = 1'b0;

      if (sel_valid[iter_var0]) begin : out_port_active
        for (iter_var1 = 0; iter_var1 < NUM_PORTS; iter_var1 = iter_var1 + 1) begin : in_port_scan
          if (sel[iter_var0] == SEL_W'(iter_var1)) begin : in_port_match
            out_flits[iter_var0] = in_flits[iter_var1];
            out_valid[iter_var0] = 1'b1;
          end : in_port_match
        end : in_port_scan
      end : out_port_active
    end : out_port_loop
  end : xbar_mux

endmodule : noc_crossbar
