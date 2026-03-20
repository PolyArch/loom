# FCC Dataflow Memory Compilation Specification

## Overview

This document specifies how FCC lowers nested-SCF memory operations into
`handshake.load`, `handshake.store`, and memory-interface operators, while
preserving the intended read/write order by explicit ctrl-done token wiring.

The core design rule is:

- data values travel on data ports
- ordering travels on a separate ctrl-done network
- every SCF region is abstracted as an `entry -> done` block
- nested regions are connected by a serial-recursive algorithm
- ctrl-done ordering is built per alias-free memory region, not as one
  mandatory global chain

The normative intent follows the older
`dsa-stack/main/lib/dsa/Transforms/SCFToHandshakeDSA` implementation,
especially:

- memory-op replacement and connection
- serial-recursive memory control construction

Related documents:

- [spec-dataflow-compilation.md](./spec-dataflow-compilation.md)
- [spec-dataflow.md](./spec-dataflow.md)
- [spec-compilation.md](./spec-compilation.md)

## Scope

This document covers:

- `memref.load` / `memref.store` replacement
- grouping accesses by root memref
- creation of memory/extmemory interface ops
- recursive ctrl-done wiring across nested `scf.if`, `scf.for`, and
  `scf.while`

This document does not define hardware memory timing. Hardware memory module
behavior is defined elsewhere.

## Memory Lowering Overview

FCC lowers memory in three conceptual layers.

### Layer 1: Replace Frontend Memory Ops

Each source `memref.load` / `memref.store` becomes a `handshake.load` /
`handshake.store`.

At this stage:

- address and data paths become explicit SSA edges
- each access is recorded with SCF-path metadata
- the memory op itself still lacks its final done-token ordering input

### Layer 2: Connect Accesses to Memory Interface Nodes

Accesses that belong to the same root memref are grouped and connected to one
memory or extmemory interface node.

At this stage:

- load address results connect to memory load-address inputs
- store data/address results connect to memory store inputs
- memory outputs connect back to the load data input
- per-access done tokens are materialized from the memory interface outputs

### Layer 3: Build Ctrl-Done Ordering Graph

The compiler then builds a second graph that carries ordering only.

At this stage:

- each memory access receives an execution-control token within its own
  ordering domain
- each memory access returns one done token within that same domain
- nested SCF structure is recursively lowered into ctrl-done composition per
  root memref group
- independent memory groups stay independent unless a later synchronization
  point explicitly joins them
- the final function return control may join group-level done tokens when the
  enclosing function as a whole must wait for all memory side effects

## Root Memref Grouping

Memory accesses must be grouped by root memref, not by the last casted or
subview value.

The grouping must strip view-like transforms such as:

- `memref.cast`
- `memref.subview`
- `memref.reinterpret_cast`
- shape-collapse or shape-expand style view rewrites

Reason:

- a single logical memory object may appear through multiple derived memref
  values
- those accesses must still share one memory ordering domain

## Memory Ordering Domains

The default ctrl-done ordering domain is one root memref group.

Normative rule:

- each alias-free root memref group gets one independent ctrl-done network
- accesses inside one such group are ordered against one another by the
  serial-recursive algorithm
- accesses in different groups must not be ordered against one another unless
  a separate synchronization rule explicitly requires it

In hardware terms, this means:

- each `handshake.memory` or `handshake.extmemory` instance normally has its
  own done-token subnet
- those done-token subnets are independent by default
- they are only joined when the program requires cross-group synchronization

This independence is not an optional optimization. It is the default semantic
shape under the alias-free memref model.

## Alias Assumption and Alias Analysis

FCC relies on the frontend to preserve alias structure as much as possible.

Normative assumption:

- distinct memrefs represent distinct logical address spaces whenever that
  separation is semantically valid

Compilation implication:

- memory alias analysis should be used to avoid collapsing unrelated memory
  regions into one overly conservative memref domain
- if two references may alias, they must remain in the same ordering domain
  unless a stronger proof separates them
- if two references are proven non-overlapping, FCC should prefer keeping them
  in separate root-memref groups so their ctrl-done networks remain
  independent

The objective is not merely better performance. It is to preserve the correct
parallelism structure of memory accesses through lowering.

## Replacing `memref.load` and `memref.store`

### `memref.load`

A `memref.load` is lowered to `handshake.load`.

Conceptually:

```text
load(addr..., data_from_mem, ctrl) -> (data_to_comp, addr_to_mem...)
```

During initial replacement:

- the address operands are connected immediately
- `data_from_mem` is temporarily left as a memory-supplied input
- `ctrl` is temporarily a placeholder

### `memref.store`

A `memref.store` is lowered to `handshake.store`.

Conceptually:

```text
store(addr..., data_from_comp, ctrl) -> (data_to_mem, addr_to_mem...)
```

During initial replacement:

- address operands are connected immediately
- data operand is connected immediately
- `ctrl` is temporarily a placeholder

### Access Metadata

Each access record must retain:

- original operation
- root memref
- access kind (`load` or `store`)
- one global ordering index
- one SCF nesting path

The SCF nesting path is required by the serial-recursive control algorithm.

## Connecting to Memory Interface Operators

After access collection, accesses are grouped by root memref and connected to
one interface operator.

### Interface Choice

The compiler may choose:

- memory-like internal memory interface
- extmemory-like external memory interface

The ordering algorithm is the same either way.

### Operand Packing Rule

Within one memory interface group:

- store operands are packed first
- load operands are packed after stores

This preserves a deterministic port layout.

### Result Routing Rule

The memory interface returns:

- load data results
- store done tokens
- load done tokens

The compiler must route:

- load data back to the matching `handshake.load`
- each done token back to the matching access record

No memory access is considered fully connected until its done token is known.

## Ctrl-Done Ordering Model

The ctrl-done graph models execution order independently from payload data.

The ctrl-done graph is built per memory ordering domain, not as one required
global graph over all memory operations in the function.

### Basic Block Abstraction

Every SCF region is abstracted as:

```text
entry_token -> region_body -> done_token
```

This is the key simplification used by the serial-recursive algorithm.

### Serial-Recursive Principle

Given a sorted list of memory accesses plus their SCF nesting paths:

- process the current SCF level from left to right
- when a child SCF region is encountered, recurse into that child
- the recursion returns one done token for the whole child block
- continue processing the parent level using that returned done token

The algorithm never reasons about raw operations alone. It reasons about
composable `entry -> done` blocks.

This recursive construction is applied independently inside each root-memref
ordering domain.

## Ordering Rule at the Same SCF Level

At the same SCF level, accesses are ordered serially unless an explicitly
allowed parallel case applies.

The default rule is:

```text
current_token -> access -> access_done -> next_token
```

This serial order is the correctness baseline.

## RAR Parallelism

Consecutive loads at the exact same SCF level may be parallelized as an
optimization when the algorithm proves there is no intervening write at that
level that must order them.

Conceptually:

- fork the entry token to the consecutive loads
- let each load execute independently
- join their done tokens before continuing

This is an optimization, not the semantic base case.

The semantic base case remains serial-recursive ordering.

RAR parallelism is only considered inside one ordering domain. It does not
justify merging two unrelated memref groups into one shared ctrl-done chain.

## `scf.if` in Memory Control

For `scf.if`, the control token is explicitly branched.

### Rule

Given parent control `%ctrl` and branch condition `%cond`:

1. split `%ctrl` by `cond_br(cond, ctrl)`
2. recursively process then-region with the true token
3. recursively process else-region with the false token
4. merge the two branch done tokens into one parent-level continuation token

### Important Constraint

The branch condition must be the branch's actual body-visible condition.

No synthetic extra loop-control event may be introduced here.

## `scf.for` in Memory Control

This is the most delicate case.

### Correct Control Source

For memory ordering inside the loop body, FCC must use the loop body's gated
control stream, not the raw stream.

That means:

- use `after_cond`
- do not use `raw_will_continue`

Reason:

- memory accesses in the body execute `N` times
- the raw stream has `N + 1` events
- wiring memory control to the raw stream introduces one extra control token
  with no matching memory payload event

### Body Block Abstraction

The loop body is abstracted as:

```text
loop_entry_ctrl -> body_memory_subgraph -> body_done
```

### Recursive Carry Form

The control loop for body-local memory ordering is modeled as a control carry:

```text
loop_ctrl = carry(after_cond, entry_ctrl, body_done_feedback)
```

Conceptually:

- the initial carry output launches the first iteration
- each true iteration feeds back its body done token
- the final false on `after_cond` exits the loop

Then:

- `loop_ctrl` drives recursive processing of the loop body
- `body_done` is split by `after_cond`
  - true branch feeds the carry feedback
  - false branch is the loop exit done token

### Normative Restriction

For body-local memory control in `scf.for`, using `raw_will_continue` is
incorrect.

The raw stream is reserved for loop-boundary structures such as iter-arg
bridging described in [spec-dataflow-compilation.md](./spec-dataflow-compilation.md).

## `scf.while` in Memory Control

`scf.while` has a before-region and an after-region.

### Rule

- before-region memory control uses the before-region condition regime
- after-region memory control uses the gated after-region condition regime
- the recursive algorithm must still treat each region as an `entry -> done`
  block

### Exit Rule

When the while condition becomes false:

- the false branch exits the loop
- there must not be one extra body-local memory-control token after exit

## Spatial Siblings and Parallel SCF Instances

If the surrounding SCF path encodes multiple independent spatial siblings,
their control subgraphs may be driven in parallel by:

- forking the parent token
- recursively processing each sibling independently
- joining sibling done tokens at the parent level

This is orthogonal to the body-local serial-recursive rule inside each sibling.

## Final Return Control

After all memory groups are processed:

- each root memref group yields one group done token
- all group done tokens are joined
- the joined token becomes the function return control token

If there are no memory accesses:

- the function return control falls back to the entry token

## Cross-Group Synchronization

Cross-group synchronization is exceptional, not the default.

FCC should only combine done tokens from different memory ordering domains
when the program semantics explicitly require a global synchronization point.

Typical examples are:

- function-level completion, when the function result is not observable until
  all memory side effects across all groups are complete
- an explicit fence-like synchronization construct
- any later construct whose semantics require multiple memory regions to have
  jointly completed before progress may continue

Outside such cases:

- each `handshake.memory` or `handshake.extmemory` done-token network should
  remain independent
- the compiler must not add artificial cross-domain dependencies

## Correctness Invariants

The following invariants are mandatory.

### Access Invariants

- every `handshake.load` and `handshake.store` must receive exactly one control
  input in the final DFG
- every access must receive exactly one memory-interface done token
- every load must receive exactly one data-from-memory connection

### Path Invariants

- every access belongs to exactly one root-memref group
- every access carries one SCF nesting path
- recursive ordering must be computed from those paths, not from textual
  locality alone
- different root-memref groups are independent ordering domains unless an
  explicit synchronization construct joins them

### Loop-Control Invariants

- body-local memory ordering in `scf.for` uses `after_cond`
- loop-boundary iter-arg bridging may use `raw_will_continue`
- these two uses must never be conflated

### Completion Invariants

- the return control token must postdominate every memory access done token
- no access may remain without its done-token consumer path

## Non-Normative Debugging Guidance

If a program finishes with:

- correct data results
- one extra latched `ctrl` in `handshake.load` or `handshake.store`
- one extra latched `cond` in downstream `handshake.cond_br`

the first thing to check is whether the memory-control chain inside a
`scf.for` body was wired to `raw_will_continue` instead of `after_cond`.
