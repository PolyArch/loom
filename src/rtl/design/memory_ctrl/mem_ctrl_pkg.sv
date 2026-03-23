// mem_ctrl_pkg.sv -- Memory controller package.
//
// Provides common types for the memory hierarchy: request/response
// structs for SPM and L2, DMA command struct, and transfer direction
// encoding.

package mem_ctrl_pkg;

  // ---------------------------------------------------------------
  // DMA transfer directions
  // ---------------------------------------------------------------
  typedef enum logic [1:0] {
    DMA_DIR_SPM_TO_L2   = 2'b00,
    DMA_DIR_L2_TO_SPM   = 2'b01,
    DMA_DIR_SPM_TO_DRAM = 2'b10,
    DMA_DIR_DRAM_TO_SPM = 2'b11
  } dma_dir_t;

  // ---------------------------------------------------------------
  // Memory request struct
  // ---------------------------------------------------------------
  typedef struct packed {
    logic [31:0] addr;
    logic [31:0] data;
    logic        wr_en;      // 1=write, 0=read
    logic [3:0]  core_id;    // requesting core
    logic [7:0]  req_id;     // request tracking
  } mem_req_t;

  // ---------------------------------------------------------------
  // Memory response struct
  // ---------------------------------------------------------------
  typedef struct packed {
    logic [31:0] data;
    logic [3:0]  core_id;
    logic [7:0]  req_id;
    logic        error;
  } mem_resp_t;

  // ---------------------------------------------------------------
  // DMA command struct
  // ---------------------------------------------------------------
  typedef struct packed {
    logic [31:0] src_addr;
    logic [31:0] dst_addr;
    logic [15:0] length;      // bytes
    logic [1:0]  direction;   // dma_dir_t encoding
    logic [3:0]  src_core_id;
    logic [3:0]  dst_core_id;
  } dma_cmd_t;

endpackage : mem_ctrl_pkg
