// fabric_cfg_if.sv -- Configuration bus interface.
//
// Word-serial configuration port with valid/ready/last handshake.
// Configuration is loaded during reset or explicit quiescent mode,
// not during active dataflow.  Matches the MMIO-based config loading
// protocol described in spec-runtime-mmio.md.

interface fabric_cfg_if #(
  parameter int unsigned ADDR_WIDTH = 16
);

  // ---------------------------------------------------------------
  // Signals
  // ---------------------------------------------------------------

  // Valid/ready handshake for config word transfer.
  logic                     cfg_valid;
  logic                     cfg_ready;

  // Config write data -- always 32 bits per spec config word model.
  logic [31:0]              cfg_wdata;

  // Config address -- width is parameterized per fabric size.
  logic [ADDR_WIDTH-1:0]    cfg_addr;

  // Last indicator -- asserted on the final word of a config burst.
  logic                     cfg_last;

  // ---------------------------------------------------------------
  // Modports
  // ---------------------------------------------------------------

  // Controller side (host or config_ctrl) drives config data.
  modport controller (
    output cfg_valid,
    output cfg_wdata,
    output cfg_addr,
    output cfg_last,
    input  cfg_ready
  );

  // Target side (fabric module) receives config data.
  modport target (
    input  cfg_valid,
    input  cfg_wdata,
    input  cfg_addr,
    input  cfg_last,
    output cfg_ready
  );

endinterface : fabric_cfg_if
