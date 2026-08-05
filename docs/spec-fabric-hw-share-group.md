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
The compiler's Canonical Dataflow specification owns `OperationSchemaId` and
its closed semantic projection. This registry references those IDs; it does
not assign aliases, copy operation attributes, or maintain a second software
operation catalog.

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

Its generated logical shape is:

```text
ImplementationFamilyDescriptor {
  family_id
  admitted_operation_schema_ids[]
  capability_params_schema_id
  typed_admission_provider_id
}
```

The descriptor remains exactly these four fields. The derived
`FabricOpSemanticFieldRelation` in
`docs/spec-fabric-reconfigurable-op.md` belongs to a
concrete resource after its enabled members, `hw_params`, ports, and
constraints are known. It is not a fifth HSG descriptor fact and is never
copied into this registry.

This is one generated descriptor, not a second runtime registry. The admission
provider interprets the declared parameter schema against a registered actor
projection and concrete physical ports. A hand-written family-shape switch,
operation-name table, or backend-local member list is a competing authority and
is forbidden.

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

An FU capability template is a separate Fabric-owned composition of concrete
nodes and edges. It references concrete `fabric.op` resources whose family,
enabled members, and `hw_params` are already fixed; it neither extends HSG
admission nor copies the family descriptor. Backend provider closure for a
selected template is derived from the implementation families of its active
operation nodes. A provider may implement those exact contracts or report
typed `Unsupported`; it cannot reinterpret the template or manufacture a
second member table.

Graph admission, simulator dispatch, and Fabric matching must all resolve the
same registered `OperationSchemaId`. An HSG descriptor may narrow where a
schema can be physically shared, but it cannot make an unregistered actor
canonical or reinterpret the schema's semantic projection.

### Initial Scalar Compute Families

`ScalarIntegerAddSub` may include `LLVMGetElementPtr` as an optional concrete
member because a stable integral GEP's final address formation can share a real
integer add/sub datapath. This is physical sharing, not semantic equivalence.
The family descriptor makes the schema eligible; a concrete `fabric.op` must
still list `LLVMGetElementPtr` in `op_list` and its typed parameters must admit
the exact pointer representation width, address-arithmetic width, address
space, and stable-integral pointer kind. An ordinary adder that omits that
member does not support GEP.

The same `LLVMGetElementPtr` schema may also belong to a dedicated address-
generation implementation family. TechMapping selects one concrete family and
does not reinterpret either family as an InstructionCore ISA. RISC-V may own
the initial LLVM DataLayout and InstructionCore ABI, but every SpatialCore
datapath is Fabric-defined.

The initial general-purpose scalar registry contains these exact family/member
relations:

| `ImplementationFamilyId` | Admitted canonical operation schemas |
| ------------------------ | ------------------------------------ |
| `ScalarIntegerAddSub` | `arith.addi`, `arith.subi`, optional `llvm.getelementptr` |
| `ScalarIntegerSaturatingAddSub` | `llvm.intr.sadd.sat`, `llvm.intr.uadd.sat`, `llvm.intr.ssub.sat`, `llvm.intr.usub.sat` |
| `ScalarIntegerCountZeros` | `math.ctlz`, `math.cttz`, poison-flagged `llvm.intr.ctlz`, poison-flagged `llvm.intr.cttz` |
| `ScalarIntegerLogic` | `arith.andi`, `arith.ori`, `arith.xori`, disjoint `llvm.or` |
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
| `ScalarFloatToInteger` | `arith.fptosi`, `arith.fptoui`, `llvm.fptosi.sat`, `llvm.fptoui.sat` |
| `ScalarIntegerMultiply` | `arith.muli` |
| `ScalarFloatMultiply` | `arith.mulf` |
| `ScalarFloatFma` | `math.fma` |

These are implementation-family identities, not FU helper names. Every family
in this table admits only scalar actor shapes; a standard operation schema may
also belong to a separately registered fixed-vector family. Basic LLVM-dialect
aliases are normalized before Canonical Dataflow and are not duplicate
members. An irreducible LLVM compute intrinsic requires its own registered
operation schema and a physically justified family admission.

Only `llvm.or` with the canonical disjoint contract is a
`ScalarIntegerLogic` member. A flag-free LLVM OR is normalized to `arith.ori`
before Canonical Dataflow. Disjointness remains exact actor semantics rather
than a hardware parameter or a runtime checker mode. The ordinary OR and
disjoint LLVM OR semantic configurations may select the same physical OR
datapath while retaining distinct canonical actor identities. In the
fixed-vector Logic family, canonical semantics marks only an affected lane as
Poison, and RTL applies the lane-local non-defined refinement without relaxing
defined sibling lanes.

The saturating and ordinary floating-to-integer schemas belong to one
`ScalarFloatToInteger` implementation family because the saturation path is
an optional extension of the same converter datapath. Family membership does
not claim that every concrete converter contains that extension. A concrete
resource enables exactly its implemented subset through `op_list`; no
`supports_saturation` parameter, second converter family, or backend-private
mode repeats that fact. The typed integer/float relation and floating behavior
profile still determine the supported format pairs and exceptional-value
contract.

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

### Initial Fixed-Vector Compute Families

Fixed-vector compute uses the same canonical operation schemas as scalar
compute, but distinct implementation families and typed shape admission. The
actor type owns rank, shape, element type, and active lane count. A concrete
capability owns admitted element domains and a positive maximum flattened
payload width; it has no independent lane-count field.

| `ImplementationFamilyId` | Admitted canonical operation schemas |
| ------------------------ | ------------------------------------ |
| `FixedVectorIntegerAddSub` | `arith.addi`, `arith.subi` |
| `FixedVectorIntegerSaturatingAddSub` | `llvm.intr.sadd.sat`, `llvm.intr.uadd.sat`, `llvm.intr.ssub.sat`, `llvm.intr.usub.sat` |
| `FixedVectorIntegerCountZeros` | fixed-vector forms of `math.ctlz`, `math.cttz`, poison-flagged `llvm.intr.ctlz`, and poison-flagged `llvm.intr.cttz` |
| `FixedVectorIntegerLogic` | fixed-vector forms of `arith.andi`, `arith.ori`, `arith.xori`, and disjoint `llvm.or` |
| `FixedVectorIntegerShift` | `arith.shli`, `arith.shrsi`, `arith.shrui` |
| `FixedVectorIntegerCompareMinMax` | `arith.cmpi`, integer min/max schemas |
| `FixedVectorValueSelect` | `arith.select` |
| `FixedVectorIntegerMultiply` | `arith.muli` |
| `FixedVectorFloatSign` | `arith.negf`, `math.absf` |
| `FixedVectorFloatAddSub` | `arith.addf`, `arith.subf` |
| `FixedVectorFloatCompareMinMax` | `arith.cmpf`, floating min/max schemas |
| `FixedVectorFloatMultiply` | `arith.mulf` |
| `FixedVectorFloatFma` | `math.fma` |
| `FixedVectorSliceAlignMerge` | `vector.extract`, `vector.insert` |
| `FixedVectorShuffle` | `vector.shuffle` |

Scalar and fixed-vector families reject one another's actor shapes even when
the flattened physical width agrees. Fixed-vector comparison results and
select conditions have the exact operand shape with `i1` elements.

`FixedVectorSliceAlignMerge` is one real position-decode, alignment, slice,
and masked-merge datapath. Its closed `FixedVectorSliceAlignMergeParams`
record owns integer-element widths, floating-element formats, positive maximum
container and slice payload widths, a nonnegative maximum dynamic-position rank,
and the admitted resolved index widths. `vector.extract` uses the selected
slice as its result. `vector.insert` preserves unselected destination bits and
merges the selected slice. A concrete resource may enable either or both
schemas through `op_list`; family membership does not manufacture an absent
extract or merge path.

`FixedVectorShuffle` is a real two-input leading-block selection and
duplication network. Its closed `FixedVectorShuffleParams` record owns
integer-element widths, floating-element formats, positive maximum operand,
result, and block payload widths, and positive maximum source- and
result-block counts. Admission derives block geometry from the exact actor
types and requires both operands, the result, and every selector domain to fit
the concrete physical ports and these capacities.

Neither parameter record owns vector shape, static position, shuffle mask, or
lane count. Those remain exact OperationSchema facts. A custom architecture
may register another physically justified family containing one of these
schemas, but FU co-location or equal flattened width is not sufficient.

### Initial Adapter And Token Families

The initial adapter and token resources are physically distinct singleton
families. Shared parameter records describe payload and fan capacity; they do
not imply circuit sharing.

| `ImplementationFamilyId` | Admitted canonical operation schema |
| ------------------------ | ----------------------------------- |
| `FixedVectorPack` | `dataflow.pack` |
| `FixedVectorUnpack` | `dataflow.unpack` |
| `FixedVectorParallelize` | `dataflow.parallelize` |
| `FixedVectorSerialize` | `dataflow.serialize` |
| `TokenConstant` | `dataflow.constant` |
| `TokenSync` | `dataflow.sync` |
| `TokenMux` | `dataflow.mux` |
| `TokenDemux` | `dataflow.demux` |

Adapter admission requires an exact fixed-vector element domain, flattened
width within capacity, and the operation schema's exact scalar/vector/mask
relation. `TokenConstant` owns a payload-capacity domain. The other token
families own payload capacity and positive maximum fan; actor arity and exact
types remain part of the canonical actor projection.

### Initial Special-Math Families

Signed quotient and signed remainder are two outputs of one signed divider
family; unsigned quotient and unsigned remainder similarly share one unsigned
divider family. Every other initial special operation is a singleton physical
family. This is deliberately conservative: FU co-location and a common
floating-point format do not prove datapath sharing.

```text
ScalarSignedIntegerDivRem   = { arith.divsi, arith.remsi }
ScalarUnsignedIntegerDivRem = { arith.divui, arith.remui }
ScalarFloatDivide           = { arith.divf }
ScalarFloatRemainder        = { arith.remf }
ScalarMathSin               = { math.sin }
ScalarMathCos               = { math.cos }
ScalarMathTan               = { math.tan }
ScalarMathSinh              = { math.sinh }
ScalarMathCosh              = { math.cosh }
ScalarMathTanh              = { math.tanh }
ScalarMathExp               = { math.exp }
ScalarMathExp2              = { math.exp2 }
ScalarMathExpM1             = { math.expm1 }
ScalarMathLog               = { math.log }
ScalarMathLog2              = { math.log2 }
ScalarMathLog10             = { math.log10 }
ScalarMathLog1p             = { math.log1p }
ScalarMathFloor             = { math.floor }
ScalarMathCeil              = { math.ceil }
ScalarMathRound             = { math.round }
ScalarMathTrunc             = { math.trunc }
ScalarMathRoundEven         = { math.roundeven }
ScalarMathSqrt              = { math.sqrt }
ScalarMathRsqrt             = { math.rsqrt }
ScalarMathErf               = { math.erf }
ScalarMathPow               = { math.powf }
```

Integer divider capabilities use the scalar integer-width record. Floating
special capabilities use the strict scalar floating-point record. Backend
provider availability is not part of family admission and does not affect a
valid Fabric artifact's identity.

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
`Unsupported` for a missing provider. A separately declared backend-ready
qualification of a builtin, independent of Fabric publication, requires
provider closure for every implementation family in the qualification's exact
realization scope.

No broader test matrix is normative. The anchor must not preserve a
member-name string bag, duplicate the registry, require exact-domain
enumeration, or make HSG identity a TechMapping candidate equivalence key.
