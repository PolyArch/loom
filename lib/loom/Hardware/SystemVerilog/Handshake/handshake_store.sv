// Handshake store: memory store adapter.
// Synchronizes addr + data + ctrl, forwards to memory.
module handshake_store #(
    parameter int ADDR_WIDTH = 32,
    parameter int DATA_WIDTH = 32
) (
    // Address from compute
    input  logic                   addr_valid,
    output logic                   addr_ready,
    input  logic [ADDR_WIDTH-1:0]  addr_data,

    // Data from compute
    input  logic                   data_valid,
    output logic                   data_ready,
    input  logic [DATA_WIDTH-1:0]  data_in,

    // Control token
    input  logic                   ctrl_valid,
    output logic                   ctrl_ready,

    // Address to memory
    output logic                   mem_addr_valid,
    input  logic                   mem_addr_ready,
    output logic [ADDR_WIDTH-1:0]  mem_addr_data,

    // Data to memory
    output logic                   mem_data_valid,
    input  logic                   mem_data_ready,
    output logic [DATA_WIDTH-1:0]  mem_data
);

  // Synchronize addr + data + ctrl (all three must be valid to fire)
  logic all_valid;
  assign all_valid = addr_valid && data_valid && ctrl_valid;

  logic mem_both_ready;
  assign mem_both_ready = mem_addr_ready && mem_data_ready;

  logic fire;
  assign fire = all_valid && mem_both_ready;

  assign mem_addr_valid = all_valid && mem_data_ready;
  assign mem_addr_data  = addr_data;

  assign mem_data_valid = all_valid && mem_addr_ready;
  assign mem_data       = data_in;

  assign addr_ready = fire;
  assign data_ready = fire;
  assign ctrl_ready = fire;

endmodule
