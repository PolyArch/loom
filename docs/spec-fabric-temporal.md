# FCC Temporal Fabric Specification

## Overview

FCC keeps temporal hardware as a first-class part of the Fabric design, even
when MVP flows focus mainly on `spatial_pe` and `spatial_sw`.

This document summarizes the temporal model without repeating every low-level
encoding detail from the rebuild plan.

## `fabric.temporal_pe`

A temporal PE contains:

- multiple `function_unit` instances
- instruction slots
- operand routing and result routing state
- optional register storage
- per-FU persistent internal configuration

Each instruction slot chooses:

- opcode
- tag match
- input mux selections
- output demux selections
- register read and write behavior

The temporal PE may execute different instructions over time, but each FU's
internal `static_mux` configuration is not reselected per instruction in the
base FCC design.

## `fabric.temporal_sw`

A temporal switch uses tag-indexed route tables.

Key properties:

- route selection depends on tag
- per-output routing remains mux-like
- arbitration and temporal correctness are part of the switch semantics

## Relationship to Spatial Components

The temporal model reuses the same broad concepts as spatial hardware:

- mux or demux controlled ingress and egress
- route-table driven switching
- explicit separation between hardware structure and runtime configuration

The main difference is that temporal hardware introduces time, tags, and
instruction state into the legality model.

## Mapper Implications

Temporal mapping must respect:

- slot capacity
- tag uniqueness where required
- register legality
- FU configuration consistency across all uses of one physical FU

These constraints are part of the mapper model rather than this summary spec.
