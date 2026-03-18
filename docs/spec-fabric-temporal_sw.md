# FCC Temporal Switch Specification

## Overview

`fabric.temporal_sw` is FCC's tag-dependent routing switch.

Placement rules:

- definitions may appear directly in the top-level module or in `fabric.module`
- inline instantiations may appear directly only in `fabric.module`
- `fabric.instance` targeting one `fabric.temporal_sw` definition may appear
  directly only in `fabric.module`

## Structural Rules

- input port count must be in `1..32`
- output port count must be in `1..32`
- every port must be `!fabric.tagged<...>`
- all ports must use exactly the same tagged type
- `num_route_table >= 1`
- if `connectivity_table` is present, it has one row per output and one column
  per input

## Runtime Configuration

`fabric.temporal_sw` uses a multi-slot route table.

Each slot contains:

- `valid`
- `tag`
- one transition payload whose length equals the number of `1` bits in the
  switch connectivity table

The important semantic rule is:

- the route table is matched by tag
- the tag is not used as a direct array index

FCC therefore uses tag-as-matching, not tag-as-indexing.

At runtime, one observed tag value may match at most one active transition
inside one `fabric.temporal_sw`.
