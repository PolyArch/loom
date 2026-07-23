# Fabric Hardware Sharing Groups

This document specifies the typed global authority for deciding which software
operation families may share one real physical implementation family. The
typed HSG registry is normative. This document specifies its semantic member
relations; one generated TableGen registry implements them, and other specs
reference family IDs instead of copying the table.

## Ownership

A Hardware Sharing Group (HSG) owns one fact: its typed software operation
families can be implemented by one actual shared datapath family. HSG legality
must be enforced by a typed verifier and realized by the Fabric-to-RTL backend.
It is not a naming convenience or a promise that unrelated operations happen
to fit in one FU container.

Registered software operation schemas remain the authority for exact actor
semantics. An HSG does not define function types, predicates, constants,
arity, or other semantic attributes. A software operation family may be legal
in more than one HSG when multiple genuine hardware implementations exist.

Each concrete `fabric.op` binds exactly one HSG implementation family. Its
`op_list` is the projection of the family members enabled by that resource;
its `hw_params`, physical ports, and typed constraints narrow those members to
the resource's actual parameterized capability. HSG membership alone neither
enables every family member nor proves that multiple activations are mutually
exclusive in resource time.

`op_list` is hardware capability, not the operation selected for one
workload. TechMapping selects one exact admitted operation and actor point;
finalization derives the corresponding typed `sw_configs`. Canonical
unconfigured Fabric stores no workload-selected member. A singleton
`op_list` therefore needs no operation-selector field, although another
semantic parameter may still require configuration.

Hardware parameters and an exact selected software configuration jointly
determine a supported configured function. Neither HSG membership nor an
operation name appearing in `op_list` permits a matcher to infer types,
attributes, arity, or any other capability not accepted by the complete typed
relation.

## Normative Family Registry

`ImplementationFamilyId` is the closed typed identity of an HSG implementation
family. One normative family descriptor owns:

* the stable family ID;
* the admitted registered `OperationSchemaId` members; and
* the family-specific typed admission rule.

The descriptor does not enumerate every exact type, value, predicate, arity,
or configuration point. Those semantics remain owned by the registered
operation schemas and are intersected with a concrete resource's parameterized
capability. One operation schema may be admitted by several implementation
families when several genuine hardware organizations implement it. Every
concrete `fabric.op` carries one explicit `ImplementationFamilyId` attribute.

One TableGen source mechanically generates the C++ enum, the MLIR enum
attribute parser and printer, and the family/member descriptors. A family may
use focused C++ verification for a complex typed admission relation; there is
no generic predicate DSL and no parallel handwritten member table. Backend
provider availability is queried by the same family ID but is not part of the
registry's semantic ownership.

### Initial Scalar Compute Families

The initial general-purpose scalar registry contains these exact family/member
relations:

| `ImplementationFamilyId` | Admitted canonical operation schemas |
| ------------------------ | ------------------------------------ |
| `ScalarIntegerAddSub` | `arith.addi`, `arith.subi` |
| `ScalarIntegerLogic` | `arith.andi`, `arith.ori`, `arith.xori` |
| `ScalarIntegerShift` | `arith.shli`, `arith.shrsi`, `arith.shrui` |
| `ScalarIntegerCompareMinMax` | `arith.cmpi`, `arith.minsi`, `arith.maxsi`, `arith.minui`, `arith.maxui` |
| `ScalarValueSelect` | `arith.select` |
| `ScalarIntegerCast` | `arith.extsi`, `arith.extui`, `arith.trunci`, `arith.index_cast`, `arith.index_castui` |
| `ScalarBitReinterpret` | `arith.bitcast` |
| `ScalarFloatSign` | `arith.negf`, `math.absf` |
| `ScalarFloatAddSub` | `arith.addf`, `arith.subf` |
| `ScalarFloatCompareMinMax` | `arith.cmpf`, `arith.minimumf`, `arith.maximumf`, `arith.minnumf`, `arith.maxnumf` |
| `ScalarFloatWidthCast` | `arith.extf`, `arith.truncf` |
| `ScalarIntegerToFloat` | `arith.sitofp`, `arith.uitofp` |
| `ScalarFloatToInteger` | `arith.fptosi`, `arith.fptoui` |
| `ScalarIntegerMultiply` | `arith.muli` |
| `ScalarFloatMultiply` | `arith.mulf` |
| `ScalarFloatFma` | `math.fma` |

These are implementation-family identities, not FU helper names. Every family
in this table admits only scalar actor shapes; a standard operation schema may
also belong to a separately registered fixed-vector family. Basic LLVM-dialect
aliases are normalized before Canonical Dataflow and are not duplicate
members. An irreducible LLVM compute intrinsic requires its own registered
operation schema and a physically justified family admission.

`ScalarValueSelect` is a runtime scalar value selector and is not the
stream-token semantics of `dataflow.mux`. `ScalarBitReinterpret` requires equal
total semantic width. Predicate, signedness, source/destination width,
floating-point policy, and other exact actor attributes remain interpreted by
the operation schema and concrete capability relation.

### Initial Loop Control Families

The initial loop-control registry contains four singleton implementation
families:

| `ImplementationFamilyId` | Admitted canonical operation schema |
| ------------------------ | ----------------------------------- |
| `LoopStream` | `dataflow.stream` |
| `LoopCarry` | `dataflow.carry` |
| `LoopInvariant` | `dataflow.invariant` |
| `LoopGate` | `dataflow.gate` |

These are physical implementation-family identities. `LoopControlFu` is an
ADG Builder helper that composes concrete resources from these families; it
is not a fifth family and does not imply circuit sharing among them.

`LoopStream` does not encode the stream step kind in its family identity.
Each concrete `LoopStream` resource fixes exactly one step kind in typed
`hw_params`; resources implementing different step kinds are distinct
`fabric.op` occurrences in the same family. Supported integer widths,
continuation predicates, ports, state, use patterns, and timing further
narrow each occurrence's capability.

The four singleton families may be replaced or supplemented by a
multi-member family only when one backend-supported circuit genuinely shares
their physical implementation while preserving every registered transition,
state, timing, and backpressure contract. FU co-location alone is not such
evidence.

## Genuine Physical Sharing

Multi-member families are legal only when one backend-supported circuit truly
shares the implementation. Examples include an ALU whose control selects add
or subtract, or a multiplier whose control selects signed or unsigned
interpretation. The exact supported types and attributes still come from the
operation schemas and the concrete resource capability.

An adder and a multiplier normally require separate datapaths. They must be
modeled as separate `fabric.op` resources even when one FU can select between
them. If those mutually exclusive resources share inputs, each shared input
requires an explicit `fabric.demux` or equivalent selector and their shared
result requires a matching `fabric.mux`:

```text
input a -> demux -> add.a / mul.a
input b -> demux -> add.b / mul.b
                    add / mul -> mux -> FU result
```

An output mux alone is insufficient. Directly using each input from both
operations is real Fabric SSA broadcast and makes both consumers participate
in token delivery and backpressure. An inactive branch cannot be treated as a
hidden drain.

## Concrete Capability Rules

Fabric verification enforces the following:

1. Every concrete `fabric.op` explicitly names exactly one registered
   `ImplementationFamilyId`.
2. Every `op_list` member belongs to that family and resolves to a registered
   software operation schema.
3. `op_list` is a subset projection of the concrete capability; it cannot
   extend capability merely because another member exists in the HSG.
4. `hw_params`, physical ports, and typed constraints form one complete,
   schema-interpretable relation for the enabled members, with no orphan or
   duplicate declaration.
5. Exact actor semantics are accepted only when the complete concrete
   capability relation supports them under ordered port correspondence.

A singleton capability remains legal. A family with one enabled member does
not need synthetic multi-operation machinery. A concrete resource does not
need to enumerate every exact semantic point in its relation.

Each family descriptor also selects one closed typed `hw_params` record
schema. Those schemas may compose reusable atoms such as an integer-width set,
floating-format set, floating-behavior profile, or cast relation, but they are
not an open property bag or predicate language. The descriptor-to-schema
binding is generated with the family registry and is not repeated by a
backend.

Canonical Fabric stores capability only. TechMapping selects a capability
template and exact actor/op/port/boundary correspondence. SpatialMapping may
select only semantic-preserving physical refinements. The finalizer derives
`sw_configs`; no HSG entry, `op_list`, or canonical `fabric.op` stores a
workload's selected operation or raw configuration bits. `ConfigurationABI`
alone owns the physical encoding of the derived configuration fields.

## Extending The Registry

Adding an HSG is a code and backend change, not a configuration escape hatch:

1. Establish that one real circuit implements the proposed typed operation
   families.
2. Add one `ImplementationFamilyId` and descriptor to the normative TableGen
   registry without creating a second member list.
3. Provide or extend a Fabric-to-RTL provider keyed by that family ID.
4. Anchor verification with one accepted shared member and one member that is
   absent from, or rejected by, the concrete capability.

Semantic Fabric verification and backend availability remain distinct. A
well-formed custom Fabric may be valid while the selected backend reports typed
`Unsupported` for a missing provider. A published builtin target must have
provider closure for every implementation family it advertises.

No broader test matrix is normative. The anchor must not preserve a
member-name string bag, duplicate the registry, require exact-domain
enumeration, or make HSG identity a TechMapping candidate equivalence key.
