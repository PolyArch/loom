// fabric_temporal_pe_scheduler.sv -- FU scheduling for temporal PE.
//
// Selects at most one FU to fire per cycle.  The selection criteria are:
//
//   1. Instruction slot matched (valid slot with matching tag -- handled
//      externally by imem; scheduler iterates over ALL valid slots).
//   2. All required operands are ready (register operands + buffer operands).
//   3. The target FU is not busy (no undrained output regs, no pipeline
//      inflight, interval counter is zero).
//   4. Interval constraint satisfied.
//
// The scheduler scans slots in ascending index order and selects the
// first slot whose conditions are all met.  This gives lower-indexed
// slots priority, matching the simulator's selectReadySlot().
//
// The scheduler is purely combinational -- it evaluates conditions and
// produces the fire decision in the same cycle.

module fabric_temporal_pe_scheduler
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_INSTR  = 4,
  parameter int unsigned NUM_FU     = 2,
  parameter int unsigned MAX_FU_IN  = 2,
  parameter int unsigned MAX_FU_OUT = 2,
  parameter int unsigned NUM_REG    = 0,
  parameter int unsigned NUM_PE_IN  = 4
)(
  // --- Per-slot decoded fields (from imem) ---
  input  logic                     slot_valid     [0:NUM_INSTR-1],
  input  logic [OPCODE_W-1:0]     slot_opcode    [0:NUM_INSTR-1],

  // Per-slot, per-operand: is this operand sourced from register?
  input  logic                     slot_operand_is_reg    [0:NUM_INSTR-1][0:MAX_FU_IN-1],

  // Per-slot input mux config
  input  logic                     slot_in_mux_discard    [0:NUM_INSTR-1][0:MAX_FU_IN-1],
  input  logic                     slot_in_mux_disconnect [0:NUM_INSTR-1][0:MAX_FU_IN-1],

  // --- Operand readiness (from operand module) ---
  // Per-slot, per-operand: buffer operand ready
  input  logic                     operand_buf_ready [0:NUM_INSTR-1][0:MAX_FU_IN-1],

  // --- Register readiness (from regfile) ---
  // Per-slot, per-operand: register operand ready
  input  logic                     operand_reg_ready [0:NUM_INSTR-1][0:MAX_FU_IN-1],

  // --- Per-FU status (from fu_slot wrappers) ---
  input  logic [NUM_FU-1:0]       fu_busy,

  // --- Number of actual inputs per FU (from parameters, generated as constants) ---
  // This could be a packed parameter, but we use a flat input for flexibility.
  // fu_num_inputs[f] = number of operands FU f actually uses.
  // For operands beyond this count, readiness is don't-care.
  input  logic [FU_IN_CNT_W-1:0]  fu_num_inputs [NUM_FU],

  // --- Scheduling result ---
  output logic                     fire_valid,
  output logic [SLOT_IDX_W-1:0]   fire_slot_idx,
  output logic [OPCODE_W-1:0]     fire_opcode,
  output logic [FU_IDX_W-1:0]     fire_fu_idx
);

  // ---------------------------------------------------------------
  // Derived widths
  // ---------------------------------------------------------------
  localparam int unsigned SLOT_IDX_W = (NUM_INSTR > 1) ? $clog2(NUM_INSTR) : 1;
  localparam int unsigned OPCODE_W   = clog2(NUM_FU);
  localparam int unsigned FU_IDX_W   = (NUM_FU > 1) ? $clog2(NUM_FU) : 1;
  localparam int unsigned FU_IN_CNT_W = $clog2(MAX_FU_IN + 1);
  localparam int unsigned EFF_OPCODE  = (OPCODE_W > 0) ? OPCODE_W : 1;

  // ---------------------------------------------------------------
  // Combinational slot scan
  // ---------------------------------------------------------------
  always_comb begin : slot_scan
    integer iter_var0;
    integer iter_var1;
    logic slot_ready;
    logic [FU_IDX_W-1:0] fu_idx;

    fire_valid    = 1'b0;
    fire_slot_idx = '0;
    fire_opcode   = '0;
    fire_fu_idx   = '0;

    for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : per_slot
      if (!fire_valid && slot_valid[iter_var0]) begin : slot_active

        // Determine which FU this slot targets
        fu_idx = slot_opcode[iter_var0][FU_IDX_W-1:0];

        // Check FU not busy
        if (!fu_busy[fu_idx]) begin : fu_not_busy

          // Check all required operands ready
          slot_ready = 1'b1;

          for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : check_operand
            // Only check operands that the FU actually uses
            if (iter_var1 < fu_num_inputs[fu_idx]) begin : in_range
              if (slot_operand_is_reg[iter_var0][iter_var1]) begin : check_reg
                // Register operand: check register readiness
                if (!operand_reg_ready[iter_var0][iter_var1]) begin : reg_not_ready
                  slot_ready = 1'b0;
                end : reg_not_ready
              end : check_reg
              else if (slot_in_mux_disconnect[iter_var0][iter_var1] ||
                       slot_in_mux_discard[iter_var0][iter_var1]) begin : skip_operand
                // Disconnected or discarded operand: always considered ready
              end : skip_operand
              else begin : check_buf
                // Buffer operand: check operand buffer readiness
                if (!operand_buf_ready[iter_var0][iter_var1]) begin : buf_not_ready
                  slot_ready = 1'b0;
                end : buf_not_ready
              end : check_buf
            end : in_range
          end : check_operand

          if (slot_ready) begin : select_slot
            fire_valid    = 1'b1;
            fire_slot_idx = iter_var0[SLOT_IDX_W-1:0];
            fire_opcode   = slot_opcode[iter_var0];
            fire_fu_idx   = fu_idx;
          end : select_slot

        end : fu_not_busy
      end : slot_active
    end : per_slot
  end : slot_scan

endmodule : fabric_temporal_pe_scheduler
