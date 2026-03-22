# LOOM FIFO Specification

## Overview

`fabric.fifo` is LOOM's single-input, single-output pipeline buffer.

Placement rules:

- named `fabric.fifo` definitions may appear directly inside the top-level
  `builtin.module` or one `fabric.module`
- inline `fabric.fifo` instantiations may appear directly only inside
  `fabric.module`
- `fabric.instance` targeting one `fabric.fifo` definition may appear directly
  only inside `fabric.module`

## Hardware Parameters

Hardware parameters live in `[]`:

- `depth`
- optional `bypassable`

Legality rules:

- `depth >= 1`
- input and output types must match
- a FIFO has exactly one input and one output

## Runtime Configuration

Runtime configuration lives in braces:

- `bypassed`

Rules:

- if `bypassable` is absent, the FIFO contributes no `config_mem` bits
- if `bypassable` is present, the FIFO contributes one runtime bit:
  - `0`: buffered FIFO behavior
  - `1`: bypass the FIFO buffer
- `bypassed` is only legal when `bypassable` is present

## Config-Memory Contribution

For one bypassable FIFO, the serialized payload is one bit:

- low bit `0`: normal buffered mode
- low bit `1`: bypass mode

Non-bypassable FIFOs contribute zero config bits.
