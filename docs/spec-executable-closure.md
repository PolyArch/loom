# Executable Closure

This document defines the persistent compiler-target and program-binary facts
needed to turn a linked C or C++ program and a complete Mapping into a
Deployment. Deployment itself is the executable closure; Loom does not add an
independent `Executable` artifact.

## Ownership

Fabric owns each HostCore and InstructionCore Architectural Contract. LLVM owns
source-module symbol, linkage, visibility, COMDAT, target-independent IR, and
ordinary link semantics. This document owns:

* exact architecture-to-codegen binding;
* target-specific InstructionCore binaries;
* host-program and runtime-registration leaves used by Deployment; and
* static initialization for Dataflow-visible logical memory roots.

SystemMapping selects AccCore occurrences but does not choose compiler targets
or binaries. Runtime validates and executes finalized choices; it never
resolves a target or substitutes a compatible binary.

## Artifact Families

```text
loom.compiler_target_binding   1.0
loom.instruction_core_binary   1.0
```

Host program bytes and Dataflow-visible static memory images are typed
Deployment leaves rather than separate artifacts. Their raw bytes are stored
by `BlobDigest` in the package closure.

## CompilerTargetBinding

```text
CompilerTargetBinding {
  version
  processor_architecture_ref :
      ArtifactReference<HostCoreOccurrenceRef>
    | ArtifactReference<InstructionCoreContextRef>
  architecture_fingerprint
  compiler_provider {
    repository_identity
    full_commit_identity
  }
  target_triple
  data_layout
  backend_abi
  object_format
  code_model
  relocation_model
  backend_cpu
  backend_features[]
  target_scope_bindings[]
  support_components[]
}

TargetScopeBinding {
  architecture_sync_scope_ref
  llvm_sync_scope_id
}

CompilerSupportComponent {
  role : StartupObject | RuntimeLibrary | BuiltinLibrary
  interface_abi_identity
  content_blob
  link_mode : Static | Dynamic
}
```

`architecture_fingerprint` is a domain-separated digest of the exact
Fabric-owned Architectural Contract. It accelerates lookup but is not a second
capability authority; resolution and import reread the exact contract and
recompute the fingerprint.

The compiler provider and target fields describe one exact LLVM codegen
configuration. A scope binding maps every required Fabric-owned synchronization
scope to one exact LLVM target scope. Support components bind exact bytes and
interface ABI once. Paths, package names, environment variables, and
unvalidated feature strings are not semantic requirements; backend feature
entries are canonical LLVM target-feature names validated by the selected
provider. `data_layout` is the exact spelling produced by reconstructing this
binding's LLVM `TargetMachine`; it is not copied from a relocatable payload.

For one exact HostCore or InstructionCore Architectural Contract and resolved
target policy, resolution
must produce exactly one binding. Zero matches are `Unsupported`; multiple
matches are an ambiguity error. There is no priority fallback. LLVM
`TargetMachine` reconstruction must produce the exact stored binding data
layout. Compatibility with the final linked LLVM module is checked by parsing
both layouts with the same pinned LLVM provider and requiring structural
`DataLayout::operator==`; equivalent spellings need not be byte-identical.
Neither artifact rewrites the other's owner field.

Microarchitectural differences that preserve one Architectural Contract may
share a binding. An ISA, endian, addressability, ABI, synchronization-scope, or
required-feature difference changes the binding or makes it incompatible.

## InstructionCoreBinary

```text
InstructionCoreBinary {
  version
  canonical_dataflow_ref
  compiler_target_binding_ref
  code_blob
  load_segments[]
  thread_entry_table[]
  runtime_imports[]
}

RuntimeImport {
  support_component_ordinal
  abi_symbol
  abi_symbol_version?
}
```

`thread_entry_table` uses canonical Dataflow thread-definition references and
binary-local entry ordinals. It does not persist raw source symbols as program
identity. Each InstructionCore binary references exactly one
InstructionCore-compatible CompilerTargetBinding. Every
unresolved import must appear in the closed typed `runtime_imports` set and
resolve to exactly one dynamic support component of that binding. ABI symbol
names are externally visible linkage semantics, not entity identity.

`load_segments[]` is the canonical parsed load manifest of `code_blob`. Each
entry stores segment ordinal, virtual address, file offset and size, memory
size, alignment, and read/write/execute permissions. Code, read-only data,
writable data, and zero-fill use the exact address, alignment, permission,
relocation, and initialization rules of the binding. The final executable
bytes are referenced by BlobDigest; segment bytes are not copied. A binary does
not embed its own artifact identity, a Deployment identity, or runtime device
addresses.

One binary may serve several SystemMapping-selected InstructionCores only when
their Architectural Contracts resolve to that exact binding. Deployment must
prove complete coverage and reject zero or multiple selected binaries for any
required thread entry.

## Host Program And Registration

Deployment contains one typed host-program leaf:

```text
HostProgramLeaf {
  compiler_target_binding_ref
  program_blob
  program_entries[]
  external_interfaces[]
  registration_table_digest
  support_component_ordinals[]
}

HostProgramEntry {
  entry_ordinal
  abi_symbol
  value_argument_types[]
  value_result_types[]
  external_interface_ordinals[]
}

HostExternalInterface {
  interface_ordinal
  kind : Value | Stream | Memory
  direction : Input | Output | InOut
  semantic_type
}
```

The referenced binding must resolve a HostCore Architectural Contract. The
host binary remains an ordinary compiler output. Its registration table is
mechanically generated from the canonical program entries and external
interfaces in the Deployment closure. It is a binary carrier, not a separate
artifact or registry authority. Static constructors, process-global scanning,
and implicit latest-registration selection are forbidden.

Entry and interface ordinals are zero-based and canonical within this leaf.
`semantic_type` is the exact registered software type: a value type, stream
element type, or logical-memory interface type according to `kind`. Direction
and kind determine which Simulation workload and runtime-input tables may use
the interface. File descriptors, environment variables, device handles, timed
external events, and arbitrary foreign-call payloads are outside the first
catalog.

The host binary cannot embed the Deployment identity because that would create
a self-reference cycle. Package loading supplies the Deployment closure and
verifies the registration-table digest before activation.

The first product surface remains ordinary C and C++ through `loom-cc` and
`loom-c++`. A stable hand-written host launch API and accelerated shared-object
loading are deferred until a concrete use requires them. Generated runtime
glue consumes the Runtime ABI without changing the source-facing program model.

## Static Logical Memory Images

```text
StaticMemoryImageLeaf {
  canonical_dataflow_ref
  logical_memory_root_ref
  layout_binding : CompilerTargetBindingRef
  size_bytes
  alignment_bytes
  permissions : ReadOnly | ReadWrite
  initialized_chunks[] {
    byte_offset
    byte_count
    blob_digest
  }
  zero_fill_ranges[]
}
```

This leaf initializes a Dataflow-visible logical memory root. It does not
duplicate code or data segments already owned by a program binary. Physical
addresses, runtime allocations, device handles, bank selection, and memory
service routes are derived from the exact Mapping and runtime admission.

The exact `CompilerTargetBindingRef` identifies the DataLayout under which the
bytes were formed. Its reconstructed DataLayout must be structurally compatible
with the final linked LLVM module and the DataLayout projected into the exact
Canonical Dataflow Program. The leaf never copies a DataLayout spelling or
invents a layout digest.

Chunks and zero-fill ranges are nonoverlapping, in bounds, canonically ordered,
and together partition `[0, size_bytes)`. `size_bytes` is positive and
`alignment_bytes` is a positive power of two. Raw bytes are referenced by
BlobDigest. A chunk's blob has exactly `byte_count` logical bytes. `ReadOnly`
is a software permission independent of whether Mapping selects local SRAM, a
manager-backed external service, or a later hardwired ROM implementation.

The final linked LLVM module is the sole initializer authority. Before raising,
the frontend mechanically projects a compiler-internal, symbol-sorted catalog
of addressable globals using the module-owned DataLayout. A definition receives
a local image only when its initializer has a complete relocation-free byte
representation. Declarations, externally initialized or thread-local storage,
non-default address spaces, relocation-bearing constants, poison, and undef
remain explicitly runtime-provided. They are not assigned guessed bytes.

For one selected `RootedGraphLaunchRef`, the exact `thread.launch` body binding
relates each imported `LogicalMemoryRootRef` to either a dynamic capability or
an `llvm.mlir.addressof` global. The symbol is an ephemeral lookup key within
that compiler invocation only. A persistent static image replaces it with the
Dataflow-owned logical-memory reference; neither the symbol nor a dense catalog
index enters Deployment identity.

## Final Link And Finalization

The frontend collects relocatable accelerator payloads only from linker-
selected objects and archive members, runs ordinary LLVM Linker/LTO semantics,
then builds the final Structured Program Candidate and Canonical Dataflow
Program. Dataflow is not a cross-object linking boundary.

Binary finalization is failure-atomic. It verifies the exact target binding,
exact reconstruction of the binding-owned data-layout spelling, structural
compatibility with the final LLVM module layout, entries, imports, segments,
blob digests, and Dataflow relation before publishing canonical bytes and
identity. It does not publish a partial binary or silently keep a Spatial
candidate on another core.

Host and InstructionCore code generation may consume the same linked program,
but they use independently exact CompilerTargetBindings. Ordinary host
`-target`, `-mcpu`, and feature options do not select an accelerator
InstructionCore target.

## Canonical Wire And Publication

`CompilerTargetBinding` and `InstructionCoreBinary` use canonical JSON semantic
bytes. Exact references use Common framing, blob fields use BlobDigest, enums
and integers use their canonical forms, and every set or table is sorted and
deduplicated by its complete typed key. Backend feature order, object section
order, source symbols that are not ABI-visible, paths, timestamps, and emitted
temporary names do not affect identity.

Finalization reconstructs the LLVM TargetMachine, validates the exact
Architectural Contract and binding-owned data layout, proves structural
compatibility with the final linked module layout, re-parses `code_blob`,
verifies the load manifest, entries, imports, and blob digests, independently
reimports the canonical JSON, then publishes atomically. HostProgramLeaf and
StaticMemoryImageLeaf are inline Deployment records and therefore acquire no
independent ArtifactIdentity.

## Anchor Verification

Anchor tests cover:

* two objects plus one archive where only linker-selected payloads reach LTO;
* zero, one, and ambiguous target-binding resolution;
* identical architecture with different compatible microarchitecture sharing
  one binding, and one ISA change producing incompatibility;
* exact binding data-layout reconstruction mismatch and structurally
  incompatible final-module layout;
* binary entry, import, segment, blob, and Dataflow-reference validation;
* complete and unique binary coverage for SystemMapping-selected thread
  entries;
* registration-table digest mismatch; and
* static logical memory overlap, bounds, permission, and layout errors.

Tests do not freeze ELF section spelling, archive layout, linker temporary
paths, host registration symbol names, or a public manual-launch API.
