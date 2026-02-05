# Dataflow Error Codes Specification

## Overview

This document defines compile-time error symbols for the Dataflow and
Handshake-software IR pipeline. These errors are raised by the Loom compiler
and do not map to hardware error codes.

All symbols in this document are **COMP_** errors.

## COMP_ (Compile-Time Errors)

| Symbol | Condition |
|--------|-----------|
| COMP_HANDSHAKE_CTRL_MULTI_MEM | A `handshake.load` or `handshake.store` control token depends on done tokens from a memory interface that is not the one associated with that access. The control chain must be rooted at the `handshake.func` start token, or depend only on done tokens produced by the same `handshake.extmemory` or `handshake.memory` used by the access. |
| COMP_DATAFLOW_CARRY_CTRL_TYPE | `dataflow.carry` operand `%d` is not `i1`. |
| COMP_DATAFLOW_CARRY_TYPE_MISMATCH | `dataflow.carry` operands `%a` and `%b` have different types, or the result `%o` type does not match `%a` and `%b`. All three must have the same type. |
| COMP_DATAFLOW_INVARIANT_CTRL_TYPE | `dataflow.invariant` operand `%d` is not `i1`. |
| COMP_DATAFLOW_INVARIANT_TYPE_MISMATCH | `dataflow.invariant` operand `%a` and result `%o` have different types. They must have the same type. |
| COMP_DATAFLOW_STREAM_OPERAND_TYPE | `dataflow.stream` operands `%start`, `%step`, or `%bound` are not `index`. |
| COMP_DATAFLOW_STREAM_INVALID_STEP_OP | `dataflow.stream` attribute `step_op` is not one of the allowed values: `+=`, `-=`, `*=`, `/=`, `<<=`, `>>=`. |
| COMP_DATAFLOW_STREAM_INVALID_STOP_COND | `dataflow.stream` attribute `stop_cond` is not one of the allowed values: `<`, `<=`, `>`, `>=`, `!=`. |
| COMP_DATAFLOW_GATE_COND_TYPE | `dataflow.gate` operand `%before_cond` or result `%after_cond` is not `i1`. |
| COMP_DATAFLOW_GATE_TYPE_MISMATCH | `dataflow.gate` operand `%before_value` and result `%after_value` have different types. They must have the same type. |
