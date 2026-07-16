# Software Function To FU Synthesis

This document specifies reverse synthesis from software function graphs to a
Fabric FU with explicit valid semantic encodings.

## Canonical Inputs And Outputs

The semantic input is a non-empty set `S` of canonical typed and attributed
software function graphs. The canonical output is Fabric topology, normalized
physical operation modes in `fabric.op.hw_params`, and a normalized
`valid_encodings` array.

The current pass may read `dataflow.subgraph` through a thin adapter because
legacy tests and passes still provide that form. The adapter immediately
constructs the shared in-memory `ConfiguredFunction` model. Synthesis,
projection, coverage, and acceptance do not use `dataflow.subgraph` as an
authority.

The central interfaces are:

```text
synthesize(config, inputs: SynthInputs) -> SynthResult
projectConfiguredFunctions(fu: FuOp) -> Array<ConfiguredFunction>
verifyCoverage(inputs, projected) -> CoverageReport
```

There is one configured-function model for both directions.

## Round-Trip Contract

Let `F = synthesize(S)` and `S' = projectConfiguredFunctions(F)`.

Every successful synthesis must satisfy `S subset-of S'`. The acceptance gate
checks every input function against an explicit Fabric encoding and records a
complete witness. A hard-coded `covered=N/N` statistic is not evidence.

Equality is valid: `S' = S`.

A strict superset is also valid when every function in `S' - S` comes from a
complete explicit legal encoding. Extra functions must not arise from
field-wise Cartesian products, unspecified don't-care values, lost tuple
correlation, or post-hoc isomorphism repair.

The coverage witness contains:

* the selected encoding index;
* software actor to physical `fabric.op` resource correspondence;
* software input port to FU input port correspondence; and
* software output port to FU output port correspondence.

Coverage failure rejects the synthesized FU.

Every covered candidate records three capability metrics:

* `encodingCount`: the number of explicit valid encodings;
* `coveredEncodingCount`: the number of distinct encodings selected by input
  coverage witnesses; and
* `extraCapabilityCount = encodingCount - coveredEncodingCount`.

Because valid encodings have pairwise distinct configured-function
projections, `extraCapabilityCount` is exactly `|S' - S|` after duplicate
inputs are collapsed by their selected encoding.

## Anchor Synthesis

Anchor synthesis aligns software actor positions across the input function
set. For each physical operation position it creates the minimum required
datapath resources and registers each observed complete typed software mode in
the owning `fabric.op.hw_params` array.

Modes that share one physical datapath must be admitted by the Hardware
Sharing Group registry. Operations from different groups require separate
physical `fabric.op` resources.

When mutually exclusive resources consume a shared software input, synthesis
inserts one explicit input demux for that input. When their results share an FU
output, synthesis inserts a matching output mux. One encoding fragment selects
all routes and the active operation mode together.

Independent physical positions may form a strict superset by combining their
complete mode fragments. Synthesis enumerates those complete resource-level
fragments directly, then retains only complete tuples whose forward projection
is type-coherent and live under the FU topology. It never unions individual
attribute fields, and it never treats an invalid cross-position tuple as a
don't-care configuration.

For each aligned physical port position, Anchor chooses the maximum software
payload width required by the input modes. A narrower software mode therefore
uses the same wider physical path without changing its exact software type.

Compilation and hardware DSE treat extra capability as an explicit metric.
Lower hardware cost remains primary. Candidates with equivalent hardware cost
prefer lower `extraCapabilityCount`, then fewer total encodings, then stable
producer order. A future profile may prefer generality, but the default is the
deterministic minimum-extra ordering.

For each generated `fabric.op`:

```text
registerHwMode(op, functionType, semanticAttributes,
               orderedInputPorts, orderedOutputPorts) -> modeIndex
```

For each generated FU configuration:

```text
encoding = { orderedOutputs, orderedResourceAssignments }
```

The complete mode is stored once on the physical operation. Encodings refer to
it by index, so `function_type` and semantic attributes do not become repeated
or competing type declarations.

## Fabric Acceptance

Before a candidate is returned, Fabric verification checks at least:

* FU topology and SSA coherence;
* Hardware Sharing Group legality;
* complete normalized `hw_params` mode structure and physical port maps;
* exact agreement between `op_list` identities and `hw_params` mode identities;
* validity of each typed and attributed software operation instance;
* semantic payload width not exceeding any selected physical path width;
* exact port-kind compatibility, with no implicit `bits`/`bits_tag` exchange;
* absence of legacy field-wise `hw_params`, selected `sw_configs`, and selected
  routing;
* normalized, live-only resource assignments;
* complete use of declared `hw_params` modes; and
* pairwise distinct configured-function projections when FU boundary identity
  is preserved.

Duplicate or non-normalized encodings are verifier errors. Synthesis must not
deduplicate projected software graphs after construction.

## Width Semantics

Software type is selected by the `hw_params` mode. Physical payload widths may
be wider. Every selected path segment must have payload width at least the
software requirement.

Physical width transitions are low-bit aligned. Widening zero-fills high bits;
narrowing truncates high bits, but a narrowing below the software semantic
width is illegal. Exact physical and software width equality is not a
capability requirement.

## Mapping Boundary

Synthesis produces canonical Fabric capability. It does not write a workload
selection into internal `sw_configs`.

TechMapping later selects one Fabric encoding and records actor/op and
boundary-port correspondence. Any transient programmed resource dictionaries
are derived from that encoding and must not be persisted back into canonical
Fabric.

## Selectable Synthesis Path

`anchor` is currently the only externally selectable strategy. Historical
`synthesize` is the sole public path that may return an accepted result. The
`anchor` producer and strategy dispatch are internal so no caller can bypass
the shared verification, coverage, and capability gate. Historical `mcs`,
`incremental`, and `incremental_random` producers were removed because they did
not emit the canonical capability contract.

`SynthConfig` is schema-closed in both YAML and TOML. Unknown keys and
sections, including removed strategy controls, fail with a source location;
they never select `anchor` by falling through to defaults.

Any future strategy must emit complete normalized `hw_params` modes and valid
encodings, then pass the same shared projection and coverage gate before it
becomes selectable. A strategy-local success flag cannot bypass that gate.

## Determinism And Parallelism

Input ordering, resource ordering, mode ordering, and encoding ordering are
deterministic. Worker-local synthesis may run concurrently, but workers do not
mutate the user's MLIR context or module. Final module insertion remains
ordered.

## Diagnostics And Statistics

Success statistics report actual input coverage, encoding count, distinct
covered encoding count, and extra capability count. Failure reports a closed
failure reason and must not retain a candidate as canonical Fabric.

Cost and search diagnostics are evaluation evidence. They rank candidates but
do not define capability, coverage, or Mapping semantics.
