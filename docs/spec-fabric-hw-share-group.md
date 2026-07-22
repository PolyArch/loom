# Fabric Hardware Sharing Groups

This document specifies the typed global authority for deciding which software
operation families may share one real physical implementation family. The
typed HSG registry is normative; this document does not duplicate its member
table.

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

Hardware parameters and an exact selected software configuration jointly
determine a supported configured function. Neither HSG membership nor an
operation name appearing in `op_list` permits a matcher to infer types,
attributes, arity, or any other capability not accepted by the complete typed
relation.

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

1. Every concrete `fabric.op` resolves exactly one registered implementation
   family.
2. Every `op_list` member belongs to that family and resolves to a registered
   software operation schema.
3. `op_list` is a subset projection of the concrete capability; it cannot
   extend capability merely because another member exists in the HSG.
4. `hw_params`, physical ports, and typed constraints form one complete,
   schema-interpretable relation for the enabled members, with no orphan or
   duplicate declaration.
5. Exact actor semantics are accepted only when the complete concrete
   capability relation supports them under ordered port correspondence.
6. The implementation family has a backend realization consistent with the
   sharing claim.

A singleton capability remains legal. A family with one enabled member does
not need synthetic multi-operation machinery. A concrete resource does not
need to enumerate every exact semantic point in its relation.

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
2. Add the family to the normative typed registry without creating a second
   member list.
3. Provide or extend the Fabric-to-RTL realization for that family.
4. Anchor verification with one accepted shared member and one member that is
   absent from, or rejected by, the concrete capability.

No broader test matrix is normative. The anchor must not preserve a
member-name string bag, duplicate the registry, require exact-domain
enumeration, or make HSG identity a TechMapping candidate equivalence key.
