// Handshake load: memory load adapter.
// Synchronizes addr + ctrl, forwards addr to memory, returns data to compute.
module handshake_load #(
    parameter int ADDR_WIDTH = 32,
    parameter int DATA_WIDTH = 32
) (
    // Address from compute
    input  logic                   addr_valid,
    output logic                   addr_ready,
    input  logic [ADDR_WIDTH-1:0]  addr_data,

    // Control token
    input  logic                   ctrl_valid,
    output logic                   ctrl_ready,

    // Data from memory
    input  logic                   mem_data_valid,
    output logic                   mem_data_ready,
    input  logic [DATA_WIDTH-1:0]  mem_data,

    // Address to memory
    output logic                   mem_addr_valid,
    input  logic                   mem_addr_ready,
    output logic [ADDR_WIDTH-1:0]  mem_addr_data,

    // Data to compute
    output logic                   comp_data_valid,
    input  logic                   comp_data_ready,
    output logic [DATA_WIDTH-1:0]  comp_data
);

  // Synchronize addr + ctrl (both must be valid to fire)
  logic sync_fire;
  assign sync_fire = addr_valid && ctrl_valid && mem_addr_ready;

  assign mem_addr_valid = addr_valid && ctrl_valid;
  assign mem_addr_data  = addr_data;

  assign addr_ready = sync_fire;
  assign ctrl_ready = sync_fire;

  // Forward memory data to compute (pass-through)
  assign comp_data_valid = mem_data_valid;
  assign comp_data       = mem_data;
  assign mem_data_ready  = comp_data_ready;

endmodule
