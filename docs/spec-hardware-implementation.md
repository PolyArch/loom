# Hardware Implementation

This document defines the immutable implementation relation for one exact
SpatialCore occurrence. Fabric remains the hardware semantic authority. A
HardwareImplementation either records the payload-free Fabric behavioral model
used by semantic runtimes or concrete state produced by RTL, ASIC, or FPGA
implementation flows. Concrete representation state cannot be recovered from
Fabric; the behavioral form is mechanically revalidated against Fabric and the
ConfigurationABI rather than copying them.

## Artifact Family

```text
loom.hardware_implementation 4.1
```

```text
HardwareImplementation {
  version
  fabric_ref
  spatial_core_occurrence_ref
  configuration_abi_ref
  representation_root
  implementation_platform_ref?
  interfaces[]
  activity_points[]
  memory_macro_bindings[]
  external_implementation_bindings[]
}
```

`fabric_ref` is an exact `loom.fabric 7.1` System root,
`spatial_core_occurrence_ref` is one exact SpatialCore occurrence in that
System, and `configuration_abi_ref` is an exact
`loom.configuration_abi 4.0` root bound to the same System. Every interface,
activity point, configuration unit, memory macro, recipe, and external binding
is confined to that occurrence. Imported Module internals use exact
occurrence-qualified physical targets. A bare Module root, an unqualified
Module-local target, a System-level resource, or another SpatialCore occurrence
cannot describe this implementation.

The represented product is the exact SpatialCore occurrence closure. It does
not claim HostCore, InstructionCore, System transport, or interconnect RTL.
Those System-level components remain architecture and simulation/modeling
facts until an independent physical provider implements them.

The artifact does not store Mapping, workload, configuration images, QoR
metrics, tool logs, report paths, or pass/fail signoff booleans.

`implementation_platform_ref` is absent for `FabricModel` and for
target-independent RTL. It is forbidden for `FabricModel`, mandatory when the
represented RTL is specialized to an ASIC technology release or FPGA ordering
code, and mandatory for GateNetlist, every ASIC physical variant, every FPGA
physical variant, and an FPGA image. Dependence on DesignWare, ChipWare,
another tool-bundled library, or an explicit user IP does not by itself create
a target manifest; that dependence is recorded by an external implementation
binding.

## Representation Root

```text
ImplementationRepresentationRoot =
    FabricModel {
      format_ref: RepresentationFormatDescriptorRef
      top: RepresentationLocator<Model>("fabric_model")
      payloads[]: exact empty array
    }
  | Rtl {
      format_ref: RepresentationFormatDescriptorRef
      top: RepresentationLocator<Module>
      payloads[]: canonical nonempty array<ImplementationPayload>
    }
  | GateNetlist {
      format_ref: RepresentationFormatDescriptorRef
      top: RepresentationLocator<Module>
      payloads[]: canonical nonempty array<ImplementationPayload>
    }
  | AsicPhysical {
      stage: Placed | Routed | Extracted
      format_ref: RepresentationFormatDescriptorRef
      top: RepresentationLocator<PhysicalObject>
      payloads[]: canonical nonempty array<ImplementationPayload>
    }
  | FpgaPhysical {
      stage: Placed | Routed
      format_ref: RepresentationFormatDescriptorRef
      top: RepresentationLocator<DeviceResource>
      payloads[]: canonical nonempty array<ImplementationPayload>
    }
  | FpgaImage {
      format_ref: RepresentationFormatDescriptorRef
      top: RepresentationLocator<DeviceResource>
      payloads[]: canonical nonempty array<ImplementationPayload>
    }

ImplementationPayload {
  role: PayloadRole
  canonical_logical_name: nonempty logical path
  blob_digest: BlobDigest
}

ImplementationPayloadRef =
  dense ordinal in canonical ImplementationPayload order
```

The stable root-variant tags are `Rtl = 0`, `GateNetlist = 1`,
`AsicPhysical = 2`, `FpgaPhysical = 3`, `FpgaImage = 4`, and
`FabricModel = 5`. ASIC stage tags are `Placed = 0`, `Routed = 1`, and
`Extracted = 2`; FPGA physical stage tags are `Placed = 0` and `Routed = 1`.
Payload-role and representation-object tags follow their displayed declaration
order below. Binary encoding uses `u32be` for every tag and `u64be` length
framing for variable byte strings and arrays. Canonical JSON uses the exact
displayed spellings and contains no aliases.

Payloads sort by `(role tag, canonical logical-name bytes, BlobDigest bytes)`.
The pair `(role, canonical_logical_name)` is unique. A logical name is a
normalized relative UTF-8 path with nonempty segments, `/` separators, and no
`.` or `..` segment; it is a namespace inside the represented state, never a
host path or an attempt output path. Dense payload refs are derived after this
ordering and a caller cannot author them.

One static typed representation-format registry owns how a payload closure is
interpreted:

```text
RepresentationFormatDescriptor {
  format_ref: RepresentationFormatDescriptorRef
  canonical locator grammar
  admissions[]: {
    exact root variant and optional stage
    exact root object kind
    exact admitted object-kind set
    exact payload role, media-type, and cardinality contract
  }
  index(
    exact root locator admitted by this descriptor,
    canonical logical payload closure read through BlobStore
  ) -> owner-typed RepresentationIndex
  lookup(RepresentationIndex, RepresentationLocator)
    -> RepresentationObjectFacts
  unresolved_external_definitions(RepresentationIndex)
    -> descriptor-owned canonical array<RepresentationLocator>
}

RepresentationObjectFacts {
  object_kind: RepresentationObjectKind
  signal_geometry?: {
    direction: Input | Output | Inout
    bit_width: positive uint64
  }
}
```

The registry identity is
`loom.hardware_representation_format`, version `2.3`. Its exact reference bytes
are `u64be(identity length) || identity bytes || u32be(major) || u32be(minor) ||
u32be(format kind)`. Existing format kinds retain their numeric meaning. A new
major version owns an incompatible indexer, object-fact, locator, or
failure-classification contract; a minor version is reserved for
backward-compatible additions. A prior-version reference is never
reinterpreted as `2.3`: there is no compatibility execution path or alias.
A canonical JSON reference is exactly the object fields `registry`, `major`,
`minor`, and `kind` in that order, with the registry string above and canonical
unsigned integers.
A MIME string, filename suffix, tool name, or caller parser cannot replace this
reference.

`format_ref` is the sole semantic identity of a descriptor and fixes every
semantic frontend option. A parser build, library revision, or compiler option
recorded by a producer is derivation provenance only; a consumer cannot use it
to select descriptor behavior. If an implementation change alters any admitted
payload, locator, object fact, unresolved-definition fact, or failure
classification, the registry version changes. A second semantic identity field
or provider-private descriptor revision is forbidden.

Registry 2.3 owns these format kinds:

| Kind | Stable spelling | Admitted root | Payload contract |
| ---: | --- | --- | --- |
| 0 | `systemverilog_rtl` | `Rtl` | one or more `RtlSource`; zero or more `GenerationConstraint` and `BlackBoxContract` |
| 1 | `structural_verilog_gate_netlist` | `GateNetlist` | one or more `Netlist`; zero or more `GenerationConstraint` and `BlackBoxContract` |
| 2 | `indexed_physical` | every `AsicPhysical`, `FpgaPhysical`, and `FpgaImage` form listed below | the exact selected physical row, including exactly one `RepresentationIndex` |
| 3 | `indexed_def_physical` | `AsicPhysical::Placed`, `AsicPhysical::Routed`, and `AsicPhysical::Extracted` | one or more structural-Verilog `Netlist`, exactly one DEF `PhysicalDatabase`, one or more `GenerationConstraint`, exactly one `RepresentationIndex`, plus the optional physical roles admitted by the selected stage |
| 4 | `fabric_model` | `FabricModel` | no payloads |

The `fabric_model` descriptor admits exactly one `Model` object whose canonical
name is `fabric_model`. Its removable index contains only that root. It parses
no payload, owns no physical or HDL object inventory, and cannot carry an
ImplementationPlatform, activity point, memory-macro binding, or external
implementation binding. The exact System and ConfigurationABI already named
by HardwareImplementation are its semantic closure. The finalizer mechanically
derives every subject-local Data, Memory, Clock, Reset, and Configuration
interface from those owners, binds each to the model root, and requires exact
equality on both finalization and import. Omitting, adding, or rebinding an
interface is invalid.

`FabricModel` is the declarative implementation relation consumed by core DFG
and CGRA runtimes. Finalizing it performs no CIRCT lowering, HDL emission,
Verilator compilation, or RTL execution. Materializing an `Rtl` root is a
separate explicit request and produces a distinct HardwareImplementation.

The two HDL descriptors use the exact media-type spellings
`text/x-systemverilog; charset=utf-8` for `RtlSource`,
`text/x-verilog; charset=utf-8` for `Netlist`,
`application/x-sdc; charset=utf-8` for `GenerationConstraint`, and
`application/vnd.loom.black-box-contract` for `BlackBoxContract`. Text payloads
use LF line endings, contain no NUL byte, and cannot depend on an ambient
include path, command-line macro, or library search order. Source or netlist
units are compiled in canonical logical-name order. An unresolved external
definition is accepted only when the complete HardwareImplementation closes it
through an exact black-box contract and external implementation binding.

The two HDL descriptors have one fixed frontend profile.
`systemverilog_rtl` uses IEEE
1800-2017 and `structural_verilog_gate_netlist` uses IEEE 1364-2005. Every
source or netlist payload is an independent preprocessing compilation unit;
all admitted declarations then participate in one elaboration library. Units
are processed in canonical logical-name order. Source-local macro definitions
are legal only within their own unit. An `include` directive is outside both
HDL descriptors. There are no caller-supplied macros, include paths,
library search paths, top-level parameter overrides, or default timescale.
Source-encoded declarations and instance parameter values remain part of the
payload bytes and are interpreted only under the selected language profile.
The frontend buffer name for each unit is exactly its canonical logical name,
so a source-location predefined macro cannot observe a host path or synthetic
temporary filename.

Their canonical locator grammar uses an unescaped HDL identifier
`[A-Za-z_][A-Za-z0-9_$]*` and a nonempty `.`-separated path of such identifiers.
A `Module` locator is one definition identifier; the representation root names
one of them as top. Instance and contained-object locators are top-rooted paths,
and a `Port` or `Pin` appends the exact terminal signal identifier. Escaped
identifiers, ambient generate-name inference, and filename-derived module names
are outside these two descriptors. Another grammar requires another exact
format reference.

Both HDL descriptors use the repository-pinned Slang frontend as their sole
parse and elaboration source. One descriptor-owned traversal derives the
removable RepresentationIndex; the Slang Compilation or AST is not the index
and is not exposed or persisted. CIRCT IR, emitted HDL, diagnostics, and a
second parser cannot supply or amend index facts. The exact top locator is an
index input and must resolve to exactly one module definition. Its ordinary
object facts contain only that top and admitted objects in its reachable
elaborated hierarchy, plus the canonical Module locators required by its
unresolved-definition inventory. An unreferenced definition neither enters the
index nor creates an external-definition requirement. The top cannot be
inferred from a filename, source order, first definition, or frontend-selected
root. A frontend-created top instance is not a second indexed object; the exact
root is classified only as `Module`.

The two HDL descriptors admit only hierarchy paths expressible by their
locator grammar.
Every occurrence is explicitly named and scalar. A generate scope may
contribute a path segment only when it is explicitly named and elaborates to
one scalar scope. Implicit generate names, generate arrays, instance arrays,
and any elaborated path requiring an index or an escaped identifier are typed
`Unsupported`. An occurrence that is legal in the selected language profile
but has no explicit source identifier, such as an unnamed generate scope or
an unnamed primitive occurrence, is `Unsupported` because admission cannot
express a locator for it. A syntactically invalid unnamed module instance is
not an admission question at all: it is an intrinsic language error that
`LanguageValid` alone classifies as `Invalid`. Cataloged contained
declarations are directly owned by a reachable module occurrence or an
admitted named scalar generate scope. Procedure, subroutine, package, class, and named-block locals,
and elaboration-only symbols such as genvars, do not enter the locator catalog.

For `systemverilog_rtl`, the exact top is a `Module`; every reachable resolved
non-top module occurrence is an `Instance`. A declared module port is a
`Port`. When the AST also exposes its backing net or variable at the same
canonical path, the `Port` is the only indexed object at that path. Every other
net declaration with one fixed positive packed-integral bit-stream width is a
`Net`. Every non-port variable with a statically sized packed-integral element
type is a `Memory` when its type has an unpacked dimension and is otherwise a
`Register`.
`Register` here means a syntactic Verilog variable; it does not claim that
synthesis will infer a flip-flop. Dynamic arrays, queues, associative arrays,
SystemVerilog interface declarations or instances, programs, checkers,
classes, unpacked net or port arrays, reference ports, and ports that combine
multiple underlying expressions are outside the initial descriptor.

For `structural_verilog_gate_netlist`, the exact top is a `Module`; every
reachable explicitly named scalar module or user-defined primitive occurrence
with declared named terminals is a `Cell`. Top-level module interface objects
are `Port`, resolved cell named interface objects are `Pin`, and all other
admitted signals are `Net`. When the AST also exposes a backing net or variable
at the same canonical path as a `Port` or `Pin`, that `Port` or `Pin` is the
only indexed object at the path. The descriptor admits module declarations,
fixed-width ports and nets, grammar-compatible static elaboration, those named
module or user-defined primitive cells, and one `GateWiringExpression`
grammar. A `GateWiringExpression` is composed only from references, integer
constants, parentheses, bit selects, part selects, concatenations, and
replications. The same grammar applies to both sides of every continuous
assignment, every net-declaration initializer, and every nonempty explicit
actual expression on a resolved module, user-defined primitive, or unresolved
module occurrence. Static parameter values, generate conditions, and
user-defined primitive truth tables are not electrical connection expressions
and do not use this grammar. Built-in gate or switch primitives, procedures,
timing controls, runtime variables or memories, behavioral subroutines, and
admitted-language connection expressions containing arithmetic, bitwise,
comparison, logical, conditional, call, cast, or streaming operators are typed
`Unsupported`. A named or unnamed built-in primitive occurrence is covered by
that primitive exclusion alone; a syntactically unnamed module instance is a
language error, not a subset violation.

One index operation evaluates two descriptor-owned logical relations over one
shared set of parse trees, followed by admission and index construction:

* `InputIntegrity` covers the payload closure contract: exact payload roles,
  cardinality, canonical order, and media types; `BlobDigest` re-reads through
  BlobStore; UTF-8, LF-only, and NUL-free text payloads; and a well-formed
  exact-root locator consistent with the descriptor's admitted root kind.
* `LanguageValid` covers whether the complete payload closure is well formed
  under the descriptor's one fixed language profile. Language facts come only
  from the pinned frontend operating in a language-validation context that
  applies no descriptor admission policy: the exact root is not preselected,
  top-level interface and reference ports stay legal, and no default
  elaboration or top-parameter rule participates. The context covers every
  unit and every definition in the closure, including definitions no exact
  root reaches. An intrinsic frontend error - malformed tokens or directives,
  syntax errors, unnamed instances, connection expressions that are illegal
  in the selected profile, undeclared identifiers, or invalid constant or
  default expressions - is a language error wherever it occurs. Descriptor
  policy never creates language errors; in particular the descriptors never
  follow `include` directives, so the frontend's report that an include could
  not be followed under that fixed policy is an admission fact, while a
  malformed `include` directive is a language error.
* `DescriptorAdmitted` covers whether a language-valid closure lies inside
  the descriptor's indexable subset: the exact-root kind and its
  elaborability under the descriptor's fixed configuration, payload and
  directive policy, the canonical locator grammar, the named scalar hierarchy
  rule, the per-descriptor structural subset, the object catalog, and the
  unknown-module policy. Admission evaluates the complete closure, so an
  excluded construct is rejected even inside a definition the exact root
  never uses. The exact root is an admission claim, never a definition of
  language validity.

The classification decision is total and order-free. Failed `InputIntegrity`
or failed `LanguageValid` is `Invalid`. The exact-root claim is evaluated
mechanically over the complete canonical payload closure: collect every
definition whose name equals the claimed exact root. Zero candidates are
`Invalid`, and more than one candidate is `Invalid`. Only exactly one
candidate proceeds to definition-kind and admission checks, so one legal
`interface` or `program` named as the exact root is `Unsupported` rather than
a missing or ambiguous root, and one `Module` continues to the
fixed-configuration elaboratability rule. A language-valid closure that fails
`DescriptorAdmitted` is `Unsupported`. Contradictory facts found while
constructing the index of an admitted closure, such as two objects with one
canonical locator, are `Invalid`. `Invalid` dominates the complete canonical
payload closure: the result never depends on payload authoring order,
frontend diagnostic order, or an admission marker found first. These fixed
examples pin the boundary:

* a legal, unique `interface` or `program` named as the exact root is
  `Unsupported`, not `Invalid`;
* a `ref` port or an interface-typed port on an otherwise legal module is
  `Unsupported`;
* a syntactically unnamed module instance such as `leaf(a);` is `Invalid`;
* a port actual that is not a legal expression in the selected profile, such
  as `.a(y = a)`, is `Invalid`, while a legal expression that only uses
  operators outside the admitted wiring grammar, such as `.a(a & b)`, is
  `Unsupported`;
* a well-formed `include` directive is `Unsupported` and is never followed;
  a malformed `include` directive is `Invalid`;
* an escaped identifier in one unit combined with a syntax error in another
  unit is `Invalid`;
* an `interface`, `program`, `checker`, or `class` that the exact root never
  uses is `Unsupported`, but an intrinsic semantic error inside an unused
  module is `Invalid`;
* a missing or ambiguous exact root is `Invalid`;
* a legal, unique module that cannot elaborate under the descriptor's fixed
  configuration, for example because it would need a top-level parameter
  override the descriptor never supplies, is `Unsupported`, while an illegal
  default or constant expression is `Invalid`;
* an unknown referenced module remains the only unresolved non-error
  condition and contributes one canonical `Module` locator with no guessed
  pin facts; and
* warnings are nonsemantic: a warning-only closure succeeds, and warnings
  never enter index facts or identity.

Every admitted `Port` or `Pin` has one fixed positive packed-integral bit-stream
width and exact direction. An unresolved module occurrence is still indexed as
an `Instance` for RTL or a `Cell` for a gate netlist, while its descriptor-owned
unresolved-definition name contributes one canonical `Module` locator.
Repeated uses of the same unresolved-definition name contribute one inventory
entry. The HDL descriptors do not guess `Pin` names, directions, or widths
for an unresolved cell. Named actual connections on such a cell are checked
only as `GateWiringExpression` values; their source spelling cannot create Pin
facts. Such facts remain unavailable until a versioned BlackBoxContract schema
owns them.

An external implementation binding closes an unresolved definition only when
its `representation_locators` include that exact `Module` locator and its
`black_box_contract_payload_ref` resolves in the same representation root.
Naming only an unresolved occurrence does not imply definition closure. Every
unresolved Module locator is closed by exactly one external implementation
binding. One binding may close several locators; overlap between bindings is
invalid.

The pinned frontend retains an unknown referenced module as the one unresolved
non-error condition. The same parse trees feed the language-validation context
and the exact-top admission elaboration. There is no second parser, no retried
language profile, no source rewriting or recovery parse, no source-word
scanning, no diagnostic-text matching, and no typed-diagnostic-code exception
table; the language-validation context is separated from admission precisely so
that descriptor policy artifacts never reach the classification. Assertions and
other SystemVerilog-only syntax under the IEEE 1364-2005 gate profile are
`Invalid`, and their illegality is decided by the frontend under the selected
profile, never by scanning source words.

Indexing reads payload bytes only through BlobStore and is pure: it cannot
execute a tool, inspect a workdir, or use a local path. The returned index is a
removable owner-typed value, not a persistent payload or second object catalog.
The HardwareImplementation finalizer uses the exact descriptor to resolve the
top and every stored locator, then independently derives expected interface
direction, width, and protocol facts from the exact Fabric and
ConfigurationABI and compares them with the indexed representation facts. The
finalizer also requires every indexed unresolved definition to be closed by an
exact black-box contract and external implementation binding, and rejects a
binding that closes no indexed definition. The format descriptor does not
redefine Fabric semantics, and the finalizer does not parse a format
independently.

The `indexed_physical` descriptor is the sole provider-neutral interpretation
of opaque ASIC and FPGA state. It never parses proprietary database, layout,
parasitic, constraint, or image bytes. It re-reads and verifies every declared
BlobDigest, then parses exactly one provider-produced `RepresentationIndex`
payload with media type
`application/vnd.loom.physical-representation-index+json`. Every other
physical payload role uses `application/octet-stream`, except
`BlackBoxContract`, which retains
`application/vnd.loom.black-box-contract`. All non-index physical payloads are
opaque to this descriptor and have no text-policy exception.

The `indexed_def_physical` descriptor is the provider-neutral, self-contained
LEF/DEF interchange form for ASIC physical state. It reuses the same physical
locator grammar and canonical `RepresentationIndex`; it does not add another
object catalog or provider identity. Its exact `PhysicalDatabase` payload is
one DEF text unit with media type `application/vnd.eda.def; charset=utf-8`.
Its `Netlist` payloads use the structural-Verilog media type and IEEE 1364-2005
profile, and its `GenerationConstraint` payloads use the SDC media type. All
three roles are UTF-8, LF-only, and NUL-free. Indexing independently validates
the retained structural netlist closure and requires its top module name and
unresolved external definitions to agree with the physical index. The DEF
parser validates the exact design name and stage-relevant physical syntax;
consumer-specific capability checks may strengthen that baseline without
assigning a private meaning to a logical filename.

The DEF descriptor exists because a routed interchange database and a
provider-native checkpoint are not substitutable inputs. A producer cannot
publish DEF bytes as opaque `indexed_physical` state when a downstream
consumer must know that they are DEF, and a consumer cannot infer DEF from a
suffix, producer name, or invocation history. Proprietary ASIC databases and
FPGA checkpoints remain `indexed_physical`; consumers that require DEF return
typed `Unsupported` for those roots.

The canonical index payload is:

```text
PhysicalRepresentationIndex {
  format_ref: exact indexed_physical or indexed_def_physical descriptor ref
  variant: AsicPhysical | FpgaPhysical | FpgaImage
  stage?: Placed | Routed | Extracted
  top: exact outer representation-root locator
  index_logical_name: exact logical name of this RepresentationIndex payload
  payloads[]: canonical nonempty array<ImplementationPayload>
  objects[]: canonical nonempty array<PhysicalRepresentationObject>
  unresolved_external_definitions[]:
    canonical array<RepresentationLocator<Module>>
}

PhysicalRepresentationObject {
  locator: RepresentationLocator
  signal_geometry?: {
    direction: Input | Output | Inout
    bit_width: positive uint64
  }
}
```

Its canonical JSON fields occur in the displayed order; `stage` is present
exactly for `AsicPhysical` and `FpgaPhysical`. Object fields are `locator` and,
when present, `signal_geometry` in that order. Signal-geometry fields are
`direction` and `bit_width` in that order. `payloads` is exactly the outer
canonical payload catalog with its one `RepresentationIndex` payload removed;
it cannot contain `RepresentationIndex`. The excluded index payload is bound
without a digest cycle by `index_logical_name`; its outer BlobDigest must equal
the digest of these exact canonical JSON bytes.

Objects sort by their locator canonical bytes and reject duplicate locators.
The exact top occurs once and has no signal geometry. `Port` and `Pin` objects
have signal geometry; every other object omits it. Unresolved definitions sort
and deduplicate by locator canonical bytes, contain only `Module` locators, and
equal exactly the set of indexed `Module` objects. The ordinary object catalog
contains only the exact top and objects rooted beneath `top` by the
descriptor's `.` separator; an unresolved one-identifier `Module` is the only
unrooted form.

The physical locator grammar reuses the unescaped HDL identifier grammar. The
top `PhysicalObject` or `DeviceResource` is one identifier. Every rooted object
appends one or more `.`-separated identifiers; a `Pin` appends a terminal after
its containing object. An ASIC admission allows `Module`, `Instance`, `Port`,
`Net`, `Register`, `Memory`, `Cell`, `Pin`, and `PhysicalObject`. An FPGA
admission allows the same logical kinds with `DeviceResource` in place of
`PhysicalObject`. A physical index cannot use a vendor-native name that is not
representable by this grammar; the provider must author the stable Loom
logical locator it publishes and use that locator consistently.

Indexing first establishes canonical outer payload state and exactly one index
role, parses the index bytes canonically, selects the exact descriptor
admission named by its variant and stage, and validates that admission's root
kind, object kinds, roles, cardinalities, and BlobStore contents. It then
requires exact equality for format ref, top, index logical name, and every
non-index payload descriptor before constructing the removable
RepresentationIndex. The HardwareImplementation finalizer additionally
requires the index variant and stage to equal the outer closed root. Missing,
stale, tampered, partial, foreign, duplicate, noncanonical, or undeclared
index, payload, or object state is `Invalid`. No provider registration can
assign a private meaning to the format kind or amend index facts.

An otherwise opaque representation that does not satisfy this complete
contract remains typed `Unsupported` rather than becoming a blob plus an
unverified top claim. New format kinds are allocated only by this registry;
provider registration cannot assign a private meaning to an existing kind.

This closed root, rather than a flat tag plus an independently interpreted
payload bag, is the sole representation authority. Its variant owns the root
locator, admitted payload roles, and required cardinalities:

| Variant | Complete allowed payload-role catalog |
| --- | --- |
| `FabricModel` | no payloads |
| `Rtl` | one or more `RtlSource`; zero or more `GenerationConstraint` and `BlackBoxContract` |
| `GateNetlist` | one or more `Netlist`; zero or more `GenerationConstraint` and `BlackBoxContract` |
| `AsicPhysical::Placed` | exactly one `RepresentationIndex`; one or more `PhysicalDatabase`; zero or more `Netlist`, `GenerationConstraint`, and `BlackBoxContract` |
| `AsicPhysical::Routed` | exactly one `RepresentationIndex`; one or more `PhysicalDatabase`; zero or more `Netlist`, `LayoutStream`, `GenerationConstraint`, and `BlackBoxContract` |
| `AsicPhysical::Extracted` | exactly one `RepresentationIndex`; one or more each of `PhysicalDatabase` and `Parasitics`; zero or more `Netlist`, `LayoutStream`, `GenerationConstraint`, and `BlackBoxContract` |
| `FpgaPhysical` | exactly one `RepresentationIndex`; one or more `PhysicalDatabase`; zero or more `GenerationConstraint` and `BlackBoxContract` |
| `FpgaImage` | exactly one each of `RepresentationIndex` and `DeviceImage` |

`GenerationConstraint` and `BlackBoxContract` payloads are present exactly
when required to reconstruct or consume that represented state. Any role or
cardinality outside the selected row is invalid. The selected format descriptor
may strengthen, but never relax, its row. Its derived index lets the
HardwareImplementation finalizer validate the root locator and every interface,
activity, external-binding, and memory-binding locator against the exact
logical payload closure. The root cannot be inferred from a filename, first
parsed module, tool default, or report.

The closed variant identifies the semantic implementation state represented
by its exact owner closure and, except for `FabricModel`, its payload closure.
It is not a linear mandatory pipeline: a selected flow may omit forms it does
not materialize. A generic stage string, flat payload catalog outside this
root, or bag of optional format fields is forbidden.

## Semantic Closure And Derivation

A HardwareImplementation is a complete self-contained description of the
represented implementation state. Its canonical root contains no parent
implementation, generator binding, or derivation edge. A consumer can import
and validate the implementation from its own exact Artifact references and
payload closure without recovering the invocation that produced it.

`InvocationManifest` is the sole owner of generation history. Its canonical
`MechanicalDerivation` or `CandidateDecision` record binds the exact typed
input Artifacts, resolved candidate-generator binding, and output
HardwareImplementation. Several valid derivations that produce identical
canonical implementation state converge on one HardwareImplementation
identity while the manifest may retain every derivation edge.

A downstream transformation may reuse payload BlobDigests from an input
implementation, but its output must enumerate every payload and exact
dependency required to reconstruct or consume the new represented state. It
cannot rely on an implicit parent closure. A generation choice that remains
relevant to the represented state must be materialized by an existing
HardwareImplementation owner such as a payload, interface, platform reference,
memory-macro binding, or external implementation binding. If none of those
owners can express a required fact, the HardwareImplementation schema requires
a specific semantic field; the generator binding cannot serve as a catch-all.

Every persisted implementation change produces another immutable
HardwareImplementation. In-place mutation is forbidden. A change to
Fabric-visible function, latency, initiation interval, capacity, buffering,
progress, reset, or ConfigurationABI is not an implementation refinement: it
requires a new Fabric and remapping.

## Semantic Payload Roles

```text
PayloadRole =
    RtlSource
  | Netlist
  | PhysicalDatabase
  | Parasitics
  | LayoutStream
  | DeviceImage
  | GenerationConstraint
  | BlackBoxContract
  | RepresentationIndex
```

The displayed order fixes stable role tags `0` through `8`; the existing tags
`0` through `7` retain their meanings. Each payload
records a closed role, canonical logical name, and BlobDigest; the selected
representation-format descriptor owns its exact media type and parser. The
artifact contains every payload required to reconstruct or consume the
represented implementation state. Backend logs, reports, waveforms, temporary
databases, and tool caches remain raw attempt bundles unless they are one of
these semantic implementation payloads.

Generated SDC or equivalent constraints use `GenerationConstraint`. Their
bytes are derived during generation from Fabric clock/reset/crossing facts, the
exact generator configuration, implementation interfaces, the target manifest
when present, and any exact external implementation contract that supplies a
required load or interface fact. The resulting payload is self-contained;
consuming it does not require the generator binding. It does not become a
timing authority or a separate constraint artifact.

## Interfaces And Activity Points

```text
ImplementationInterface {
  semantic_ref: ImplementationInterfaceSemanticRef
  representation_locator
  device_pin_ref?
}

ActivityPoint {
  representation_locator
  semantic_fabric_ref?
}

ImplementationInterfaceSemanticRef =
    Data(FabricSpatialAttachmentEndpointRef on the Transport plane)
  | Memory(FabricSpatialAttachmentEndpointRef on the Memory plane)
  | Clock(HardwareDomainRef whose contract kind is Clock)
  | Reset(HardwareDomainRef whose contract kind is Reset)
  | Configuration(ProgrammingUnitRef in the exact ConfigurationABI)
  | ExternalProtocol(ExternalBoundaryRef in the exact System)

RepresentationLocator {
  object_kind: RepresentationObjectKind
  canonical_name
}

RepresentationObjectKind =
    Module
  | Instance
  | Port
  | Net
  | Register
  | Memory
  | Cell
  | Pin
  | PhysicalObject
  | DeviceResource
  | Model
```

The displayed order fixes stable representation-object tags `0` through `10`.
Every locator encodes its `u32be` object-kind tag followed by the
`u64be`-length-framed canonical-name bytes. Locator arrays are sorted by these
canonical bytes and reject duplicates.

The displayed interface-semantic alternatives have stable tags `0` through
`5`. The tag is the interface role; no separately authored role or interface
key can disagree with it. Each nested reference retains its owner-defined
canonical bytes. `Configuration` contains the exact cross-Artifact
`ProgrammingUnitRef` defined by `docs/spec-configuration-deployment.md`; its
ConfigurationABI Artifact reference must equal `configuration_abi_ref`.
Canonical JSON spells the union as an object with exactly `kind` and `target`:
`kind` is the displayed alternative name and `target` is the lowercase
hexadecimal spelling of that alternative's complete nested canonical bytes.
No alternative has a second JSON shape or alias.

The interface catalog binds declared Fabric-visible boundaries, clocks,
resets, configuration transports, memories, and external protocols to exact
implementation locators. The HardwareImplementation finalizer validates every
declared semantic reference against its exact System or ConfigurationABI and
validates every locator against the representation index. It does not invent
a universal port or protocol-signal inventory. An invocation or runtime
consumer declares the semantic interfaces it requires and rejects an absent or
ambiguous match before execution. This keeps protocol decomposition with the
consumer or provider that owns it instead of making HardwareImplementation a
second AXI, JTAG, MMIO, memory, or external-protocol schema.

For `FabricModel`, the catalog is not caller-declared: the finalizer derives
the complete subject-local interface closure from the exact System and ABI.
All records use the sole `Model("fabric_model")` locator because a behavioral
model has no separately materialized signal objects. RuntimePlatformBinding
still binds each typed semantic interface independently; sharing the model
locator does not merge their identities or endpoint contracts.

Several `Configuration(ProgrammingUnitRef)` records may intentionally name the
same top-module locator when one shared transport serves their exact
occurrence-local units. The selected provider contract owns the transport
signal decomposition, while the shared `ConfigurationTransportLayout`
derivation owned by the Configuration/RTL contracts maps each semantic unit to
its transport-local window. HardwareImplementation does not copy that address
table or invent a Core ID. A missing semantic unit record, a foreign unit, or a
provider projection that disagrees with the shared derivation is invalid.

The activity catalog is the sole implementation-local source for RTL,
netlist, physical, and FPGA activity references used by simulation or
Evaluation. Activity points are declared capabilities, not a universally
derivable catalog. A consumer that requires one names its exact owner-local
reference and fails before execution if it is absent.

The enclosing representation and exact format descriptor give every locator
its representation-local interpretation; a locator therefore does not repeat
a representation tag.
`canonical_name` is the stable name within that exact represented state, not a
filesystem path, report path, tool query, or Fabric entity name. The closed
object kind prevents a port, net, cell, pin, physical object, and device
resource from becoming interchangeable strings. A locator kind incompatible
with the enclosing representation is invalid.

Locators do not alter Fabric or Mapping identity. `device_pin_ref` is valid
only for an FPGA representation with an exact FPGA target manifest.

Interfaces sort by their complete canonical records and activity points sort
by `(representation_locator, optional semantic_fabric_ref)`. Both catalogs
reject duplicate records. One representation locator may implement several
semantic references, and one semantic reference may require several locators;
the exact consumer contract owns that grouping. Their dense ordinals are
derived only after sorting; no caller-authored interface or activity ID enters
identity.

The HardwareImplementation 4.1 owner-local reference catalog is:

```text
0  HardwareImplementationInterfaceRef
1  HardwareImplementationActivityPointRef
2  ExternalImplementationBindingRef
```

Each local payload is one `u64be` dense ordinal into the corresponding
canonical catalog. A complete cross-artifact reference uses the Common exact
HardwareImplementation Artifact identity plus this owner-local kind and
payload. Strict decoding rejects an unknown kind, out-of-range ordinal,
noncanonical catalog, or a target whose enclosing representation does not
admit its locator.

## Memory And External Bindings

```text
ExternalDependencyIdentity =
    ExplicitFile {
      content_sha256
    }
  | ToolBundledResource {
      stable_provider_build_identity
      resource_key
    }

ExternalInputBinding {
  provider_input_slot_ref
  dependency_identity: ExternalDependencyIdentity
}

ExternalImplementationBinding {
  provider_contract_ref
  external_inputs: canonical nonempty catalog<ExternalInputBinding>
  fabric_resource_refs: canonical set<FabricPhysicalOccurrenceOwnerRef>
  representation_locators[]
  black_box_contract_payload_ref?
}

ExternalImplementationBindingRef =
  dense owner-local ordinal in canonical binding-key order

MemoryMacroBinding {
  fabric_memory_ref: FabricPhysicalOccurrenceOwnerRef refined to memory
  external_implementation_binding_ref
  representation_locator
}
```

An explicit-file identity is the SHA-256 fingerprint of exactly one ordinary
file selected through a provider-owned typed input slot. It is not a BlobDigest
claim and does not require Loom to copy the file into BlobStore. A tool-bundled
identity combines the stable provider build selected by the semantic binding
with one exact provider resource key. A display version alone is invalid when
the provider requires a stronger build identity.

The exact provider contract is the sole owner of input-slot identity, role,
cardinality, and compatibility. An external implementation binding records the
closed slot-to-dependency relation required by its represented implementation
state. A memory macro may therefore bind distinct logical, timing, physical,
and layout files without collapsing them into one directory or one digest.
A representation includes only the slots required to reconstruct or consume
that state; a later state closes its own required set rather than inheriting an
implicit earlier binding.

An external binding has no caller-authored ID. Its canonical key is the exact
tuple of provider contract, external-input catalog, occurrence-qualified Fabric
relations, representation locators, and optional black-box payload reference.
Finalization sorts and deduplicates complete keys, then assigns dense
owner-local ordinals. `MemoryMacroBinding` and every other internal reference
use only that derived ordinal. Authoring order, a display label, and a stale or
sparse supplied number cannot become binding identity.

Within each binding, `fabric_resource_refs` and `representation_locators` are
canonical sorted-unique arrays. `black_box_contract_payload_ref`, when present,
must resolve to an `ImplementationPayloadRef` in the same root whose role is
exactly `BlackBoxContract`. A memory-macro locator must pass the same selected
representation-format index/lookup and HardwareImplementation finalizer
cross-check as every other locator.

`external_implementation_bindings` cover vendor arithmetic libraries, FPGA
primitives or configured IP, fixed or generated memory macros, encrypted or
black-box user IP, and technology libraries instantiated by the represented
state. The provider contract owns each dependency's typed interpretation and
compatibility rules. The HardwareImplementation owner owns only the closed
slot-to-identity relation, exact Fabric relations, locators, and optional
black-box payload relation shown above. Paths, filenames as semantic roles,
and free-form property maps are forbidden.

`memory_macro_bindings` map exact Fabric memory occurrences to one compatible
external implementation binding and representation locator. Fabric owns the
required memory semantics. The provider contract owns the offered macro
contract and exact external view slots. ImplementationPlatform does not contain
a macro library or its files.

Synthesizable user source that is incorporated into the represented RTL is an
`RtlSource` payload rather than an external file dependency. An encrypted or
otherwise nonmaterialized implementation remains an external binding with an
exact `BlackBoxContract`. No binding permits a missing dependency to masquerade
as a complete implementation.

These bindings are downstream `HardwareImplementation` facts. They are not
Fabric `ImplementationInput` dependencies and cannot be used to make that
reserved-unavailable `loom.fabric 7.x` role legal. An Interconnect
Implementation remains a separate System-level product; provider-owned
external state for the selected SpatialCore occurrence is selected and
validated here.

An implementation provider may report `Unsupported` when an otherwise valid
Fabric resource lacks an implementation. It cannot emit a substituted,
truncated, or placeholder implementation.

## Physical Design Boundaries

The first version supports one always-on power state. Power gating, isolation,
retention, DVFS, partial reconfiguration, DFT insertion, ATPG, fault injection,
and general reliability policy are explicitly deferred until they have an
independently observable contract.

Floorplan, placement, routing, and tool-control choices are typed generator
configuration and resulting implementation payloads. Loom does not define a
global floorplan or vendor-command DSL. FPGA uses the same immutable
implementation family as ASIC; the first version produces a static full-device
image and does not claim partial-reconfiguration support.

ASIC physical implementation follows the same hierarchy as synthesis. Leaf
operation, switch, memory, FIFO, and queue blocks publish exact physical views
and boundary timing abstractions. SpatialCore placement and routing assemble
those views, route only the owning hierarchy's interconnect, and perform
top-level timing and physical-consistency checks. A provider must not flatten
all leaf logic into one SpatialCore physical attempt as a substitute for this
composition. Reused physical blocks are keyed by complete representation,
platform, corner, constraint, and algorithm identities, never by path,
application name, object address, or occurrence ordinal alone.

Timing, power, area, thermal, DRC, and other observations are
EvaluationEvidence over the exact HardwareImplementation. Negative slack or a
physical violation may be reported for a completed implementation; a tool
failure that produces no coherent represented state publishes no
HardwareImplementation.

Timing closure follows semantic ownership. Gate sizing, placement, or another
implementation-only choice creates another HardwareImplementation when it
changes materialized semantic state. `InvocationManifest` records the exact
input and generation decision. Selecting a Fabric-declared bypass, buffer, or
latency refinement changes Mapping and its configuration. Inserting state or
changing latency, initiation interval, or recurrence behavior outside such a
declared refinement creates a new Fabric candidate and requires remapping.
Central DSE composes the resulting candidates and EvaluationEvidence; an EDA
adapter cannot mutate any of those owners in place.

## Finalization

Finalization verifies the exact Fabric System, SpatialCore occurrence,
ConfigurationABI, optional target manifest, closed representation root, exact format
descriptor, payload roles, BlobStore bytes and digests, interfaces, activity
points, memory-macro bindings, and external bindings. It resolves every
provider contract, validates each external dependency identity, verifies every
Fabric-resource relation, builds the selected descriptor's pure index over the
exact logical payload closure, resolves and cross-checks the complete locator
set, and rejects an external module without its required black-box contract.
It also verifies that the represented state has no implicit dependency on a
parent implementation or generator invocation. Canonical ordering uses typed
semantic keys. Filesystem
paths, mtimes, tool invocation order, generator search history, and reports do
not enter identity.

Canonical semantic bytes are one canonical JSON root containing exact Artifact
references and BlobDigest payload references. Closed variants and typed keys
use their registered canonical spelling; sets and catalogs are sorted and
deduplicated by complete semantic key. Backend-native manifests are payloads,
not an alternate HardwareImplementation root.

The artifact is published only after every required payload and binding is
available and independently re-readable. Partial manifests and path-based
success markers are invalid. Finalization independently reimports and verifies
the canonical root before atomic publication.

For any downstream producer, the authoritative implementation bytes are only
the logical bytes returned by `BlobStore` for the `BlobDigest` references in
the exact input's `representation_root`. The producer rehashes those bytes
before materialization. A previous invocation directory, declared output path,
report path, vendor database path, or caller-supplied duplicate source is not a
production input. Explicit PDK, cell, macro, and user-IP dependencies retain
their existing external-binding identities; this rule does not import their
installation trees into BlobStore.

## Anchor Verification

Anchor tests cover:

* manifest edges for `Fabric + ConfigurationABI -> H_rtl` and
  `H_rtl -> H_next`, where `H_rtl` is the derived RTL implementation and
  `H_next` is its direct derived successor, retaining exact typed inputs and
  resolved generator bindings while neither output copies parent or generator
  lineage into its Artifact body;
* two distinct derivations with identical canonical implementation state
  converging on one HardwareImplementation identity;
* a recipe-only structural change changing HardwareImplementation identity
  exactly when it changes a materialized payload or binding, without changing
  Fabric or ConfigurationABI identity;
* rejection of a derived implementation that depends on an implicit parent
  payload;
* a Fabric-visible timing or capacity change being rejected as an
  implementation refinement;
* missing, duplicate, wrong-role, or corrupt payloads;
* stable root, stage, payload-role, and object-kind tags, payload logical-name
  normalization, dense local-reference round trips, and one known canonical
  root vector;
* variant-specific representation-root locator and payload cardinality, with a
  flat or inferred top rejected;
* exact-top-dependent reachability, independent compilation-unit macro scope,
  canonical logical source names, rejection of ambient frontend inputs, and
  warning-invariant index results;
* the fixed classification matrix: intrinsic language errors are `Invalid`
  anywhere in the closure and dominate admission rejections in either payload
  order, a non-module or non-elaboratable exact root is `Unsupported`, and a
  missing or ambiguous exact root is `Invalid`;
* RTL Port precedence over a backing net or variable, syntactic
  Register-versus-Memory classification, and rejection of unsupported dynamic
  or interface-like objects;
* explicitly named scalar hierarchy and generate scopes, with syntactically
  unnamed occurrences classified `Invalid` and arrayed, implicit-name,
  built-in-primitive, and non-grammar paths typed `Unsupported`;
* gate module or named user-defined-primitive Cell and Pin classification,
  Port and Pin precedence over backing objects, wiring-only continuous
  assignments, and rejection of behavioral or operator-bearing structural
  input;
* top-reachable unresolved Module inventory, duplicate-use convergence, exact
  one-binding closure without overlap, and no guessed Pin facts for an
  unresolved cell;
* missing representation-format providers, wrong-format payloads, locators
  absent from the exact logical representation, and an opaque database without
  its descriptor-required canonical index;
* physical index encode/decode/re-encode determinism and rejection of a
  missing, stale, tampered, partial, foreign, duplicate, noncanonical, or
  undeclared index, payload, object, or top claim;
* required interface, activity-point, memory-macro, and configuration-
  transport coverage;
* explicit-file and tool-bundled external identities producing distinct
  bindings without importing their installation trees;
* authoring-order-independent dense external-binding references and rejection
  of a supplied, sparse, duplicate, or stale binding ordinal;
* a memory macro selecting one exact external binding and rejection of a
  platform-owned macro-file lookup;
* ASIC and FPGA representation variants under the same family; and
* downstream materialization accepting only BlobStore-verified bytes from the
  exact representation root and rejecting a substituted source or work path;
* completed adverse timing evidence versus a tool failure that publishes no
  HardwareImplementation.

Tests do not freeze vendor report text, Tcl formatting, database directory
layout, every EDA tool, or a large format-conversion matrix.
