#ifndef FABRIC_OPS_TD
#define FABRIC_OPS_TD

include "Fabric/IR/FabricDialect.td"
include "Fabric/IR/FabricAttrs.td"
include "Fabric/IR/FabricEnums.td"
include "Fabric/IR/FabricTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/Properties.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Fabric_TypeProp : Property<"::mlir::Type", "MLIR type"> {
  let convertFromAttribute = [{
    auto typeAttr = ::llvm::dyn_cast_if_present<::mlir::TypeAttr>($_attr);
    if (!typeAttr)
      return $_diag() << "expected type attribute";
    $_storage = typeAttr.getValue();
    return ::mlir::success();
  }];
  let convertToAttribute = [{
    return ::mlir::TypeAttr::get($_storage);
  }];
  let parser = [{
    if (::mlir::failed($_parser.parseType($_storage)))
      return ::mlir::failure();
  }];
  let printer = "$_printer.printType($_storage)";
  let readFromMlirBytecode = [{
    if (::mlir::failed($_reader.readType($_storage)))
      return ::mlir::failure();
  }];
  let writeToMlirBytecode = "$_writer.writeType($_storage)";
  let hashProperty = "::llvm::hash_value($_storage.getAsOpaquePointer())";
}

def Fabric_InnerInputTypesProp
    : ArrayProp<Fabric_TypeProp, "inner input type array"> {
  let defaultValue = "::llvm::ArrayRef<::mlir::Type>{}";
  let storageTypeValueOverride = "::llvm::SmallVector<::mlir::Type>{}";
}

def Fabric_MuxOp : Fabric_Op<"mux", [Pure, SameOperandsAndResultType]> {
  let summary = "fabric multiplexer with handshake";
  let description = [{
    Selects one of N (N > 1) inputs onto the single output. All inputs and the
    output share one `!fabric.bits<W>` type. No hardware parameters (the port
    width is fully described by the shared operand/result type).

    Software parameters (either all set or all unset):
      * `sel`        : i32, 0 <= sel < N (forced 0 when `disconnect` is true)
      * `discard`    : bool
      * `disconnect` : bool

    Unset means the op describes pure hardware (not programmed). Programmed
    combinations must satisfy: !(discard && disconnect).
  }];

  let arguments = (ins Variadic<Fabric_BitsType>:$inputs,
      OptionalAttr<I32Attr>:$sel, OptionalAttr<BoolAttr>:$discard,
      OptionalAttr<BoolAttr>:$disconnect);
  let results = (outs Fabric_BitsType:$output);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_DemuxOp : Fabric_Op<"demux", [Pure, SameOperandsAndResultType]> {
  let summary = "fabric demultiplexer with handshake";
  let description = [{
    Single input, N outputs (N > 1). All input/output ports share one
    `!fabric.bits<W>` type. No hardware parameters (the port width is fully
    described by the shared operand/result type). The op is the input/output
    dual of fabric.mux.

    Software parameters (either all set or all unset):
      * `sel`        : i32, 0 <= sel < N (forced 0 when `disconnect` is true)
      * `discard`    : bool
      * `disconnect` : bool

    Unset means the op describes pure hardware (not programmed). When
    programmed, `discard && disconnect` is forbidden. Untying valid/ready
    on the unselected outputs:

      * pure hardware: input.ready=0, all outputs.valid=0 (= disconnect)
      * sel only    : input <-> outputs[sel] passthrough; other outputs.valid=0
      * discard=T   : input.ready=1 (input drained); all outputs.valid=0
      * disconnect=T: input.ready=0; all outputs.valid=0
  }];

  let arguments = (ins Fabric_BitsType:$input, OptionalAttr<I32Attr>:$sel,
      OptionalAttr<BoolAttr>:$discard, OptionalAttr<BoolAttr>:$disconnect);
  let results = (outs Variadic<Fabric_BitsType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_FifoOp : Fabric_Op<"fifo", [Pure]> {
  let summary = "fabric FIFO with handshake";
  let description = [{
    Single-input single-output FIFO. Input and output share one Fabric port
    kind. Their widths may differ under the module-level low-bit-aligned
    truncation or zero-extension rule; the output type is the FIFO's internal
    storage width.

    Hardware parameters (always present, printed in `[...]`):
      * `max_depth`   : i32, must be > 0 (physical FIFO depth)
      * `bypassable`  : bool (whether hardware supports combinational bypass)

    Software parameter (only valid when `bypassable` is true, printed in `{}`):
      * `bypassed`    : bool (when true, the FIFO is configured in bypass mode;
                       data/valid/ready become a combinational passthrough)

    Unset `bypassed` describes pure hardware (not programmed). Setting
    `bypassed` when `bypassable` is false is illegal.
  }];

  let arguments = (ins Fabric_AnyType:$input, I32Attr:$max_depth,
      BoolAttr:$bypassable, OptionalAttr<BoolAttr>:$bypassed);
  let results = (outs Fabric_AnyType:$output);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_OpOp : Fabric_Op<"op", []> {
  let summary =
      "fabric module instance: a configurable hardware area that can implement "
      "one of a set of software ops";
  let description = [{
    Models one physical operation datapath. Its explicit implementation family
    selects the generated member relation and typed capability schema. The
    required `op_list` is the enabled subset exposed by this occurrence.
    Hardware ports are `fabric.bits<N>` only.

    Attributes:
      * `implementation_family`: one explicit generated HSG family identity.
      * `op_list`    : checked array of `FlatSymbolRefAttr`, e.g.
                       `[@arith.addi, @arith.subi]`.
      * `hw_params`  : the one closed typed capability record selected by the
                       generated implementation-family descriptor. The
                       verifier rejects unknown fields, malformed domains,
                       and a record of another family schema.
  }];

  let arguments = (ins Variadic<Fabric_BitsType>:$inputs,
      OptionalAttr<Fabric_ImplementationFamilyAttr>:$implementation_family,
      ArrayAttr:$op_list, DictionaryAttr:$hw_params);
  let results = (outs Variadic<Fabric_BitsType>:$outputs);

  let assemblyFormat = [{
    $op_list `(` $inputs `)` attr-dict
      `:` functional-type($inputs, $outputs)
  }];
  let hasVerifier = 1;
}

def Fabric_ModulePortType
    : AnyTypeOf<[Fabric_BitsType, Fabric_BitsTagType, AnyMemRef]>;

def Fabric_ModuleOp
    : Fabric_Op<"module", [IsolatedFromAbove,
                           SingleBlockImplicitTerminator<"YieldOp">, Symbol,
                           SymbolTable,
                           DeclareOpInterfaceMethods<RegionKindInterface>]> {
  let summary = "fabric module: top-level container of the fabric dialect";
  let description = [{
    Top-level container for the fabric dialect. Carries a required
    `sym_name`. The body is a single graph-region block whose
    terminator is `fabric.yield`.

    Module inputs are declared as the body's entry-block arguments
    (the op itself has zero SSA operands, like `func.func`). Module
    outputs are declared as result types on the op signature; the
    `fabric.yield` terminator supplies their values.

    Allowed input/output types: `!fabric.bits<W>`,
    `!fabric.bits_tag<W, T>`, `memref<...>`. No other MLIR types are
    accepted.

    The body accepts `fabric.pe`, `fabric.switch`, `fabric.mem`,
    `fabric.fifo`, nested `fabric.module`, `fabric.instantiate`, and
    `fabric.boundary`, plus the implicit `fabric.yield` terminator. See
    `docs/spec-fabric-module.md` for the full rules.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      // Optional exact module-scoped semantic values. Absence remains
      // absence during elaboration; no Fabric-local default is applied.
      OptionalAttr<I32Attr>:$loom_addr_bits,
      OptionalAttr<I32Attr>:$loom_mem_bus_width,
      // Finalization-owned canonical Module domain relation. Authoring roots
      // omit these carriers; every canonical root materializes both arrays.
      OptionalAttr<ArrayAttr>:$domain_slots,
      OptionalAttr<ArrayAttr>:$domain_assignments);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_PeOp
    : Fabric_Op<"pe", [IsolatedFromAbove, SingleBlock, NoTerminator,
                       DeclareOpInterfaceMethods<RegionKindInterface>,
                       DeclareOpInterfaceMethods<Symbol, ["isOptionalSymbol"]>,
                       HasParent<"::fabric::ModuleOp">, SymbolTable]> {
  let summary =
      "fabric PE: a container of one or more fabric.fu ops with a schedule "
      "predicate (spatial or temporal)";
  let description = [{
    Processing element. Holds one or more `fabric.fu` ops in its body and
    carries a mandatory `schedule` predicate (`spatial` | `temporal`) that
    declares how the contained FUs are time-shared.

    The op exists in two disjoint syntactic forms by `sym_name` presence:

    Anonymous form (definition + use combined). Carries variadic SSA
    operands and variadic SSA results. The body has no terminator and
    inner FUs are not SSA-wired to the PE results.

      %r:2 = fabric.pe [spatial]
                 (%pa = %a : !fabric.bits<W>,
                  %pb = %b : !fabric.bits<W>)
                 -> (!fabric.bits<W>, !fabric.bits<W>) {
        fabric.fu(...) -> ...
      }

    Named template form (declaration only). Has zero SSA operands and zero
    SSA results in the host scope; `function_type` alone owns the logical
    port signature. The body's entry block carries the input port types as
    block arguments, and the body closes with a zero-operand
    `fabric.yield` signature terminator: it never restates the result
    ports, whose internal sources are chosen by the configured output
    selectors. Actual usage of a named PE goes through
    `fabric.instantiate`, which derives its result arity and types from
    `function_type`.

      fabric.pe @ALU [spatial]
                (!fabric.bits<W>, !fabric.bits<W>) -> (!fabric.bits<W>) {
      ^bb0(%pa: !fabric.bits<W>, %pb: !fabric.bits<W>):
        fabric.fu(%fa = %pa : !fabric.bits<W>) -> !fabric.bits<W> { ... }
        fabric.yield
      }

    A temporal PE uses tagged boundary ports and carries the hardware and
    software configuration attributes declared below. Its verifier checks
    configuration completeness, instruction encoding, selector bounds, and
    FU compatibility. See `docs/spec-fabric-pe-temporal.md` for the complete
    temporal constraints.

    Constraints (spatial, both forms):
      * `K = numInputs() >= 1` and `L = numOutputs() >= 1`. For the
        anonymous form `K`/`L` are the operand/result counts; for the
        named form they are the function-type input/output counts.
      * Every PE port is `!fabric.bits<W>` with the same `W`. Mixing
        widths or non-`bits` fabric types on the PE boundary is rejected.
      * The body holds `fabric.fu` and `fabric.instantiate` ops and at
        least one such compute member. No `fabric.op`, `fabric.mux`,
        `fabric.demux`, `fabric.fifo`, or `fabric.pe` may appear
        directly in the body. `fabric.yield` may appear only in the
        named form, where it must carry zero operands.
      * For every inner FU `f`, `f.numInputs() <= K` and
        `f.numOutputs() <= L`. Every FU outer input port and every FU
        outer output port has type `!fabric.bits<W>` matching the PE's
        `W`. Inner FU block-arg types may be narrower
        (`!fabric.bits<F>` with `F <= W`), in which case the FU
        boundary truncates the high `W - F` bits.
      * The op must appear inside a `fabric.module` body.
  }];

  let arguments = (ins OptionalAttr<SymbolNameAttr>:$sym_name,
      OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
      Fabric_ScheduleAttr:$schedule, Variadic<Fabric_AnyType>:$inputs,
      // Temporal-only hardware parameters. All must be absent on a
      // spatial PE; presence/absence rules per-attribute are enforced
      // by the verifier (see docs/spec-fabric-pe-temporal.md).
      OptionalAttr<I32Attr>:$tag_width, OptionalAttr<I32Attr>:$num_instruction,
      OptionalAttr<I32Attr>:$num_reg_fifo,
      OptionalAttr<I32Attr>:$reg_fifo_depth,
      OptionalAttr<I32Attr>:$reg_fifo_ports,
      OptionalAttr<Fabric_FuConfigModeAttr>:$fu_config_mode,
      OptionalAttr<Fabric_OperandBufferModeAttr>:$operand_buffer_mode,
      OptionalAttr<I32Attr>:$operand_buffer_size);
  let results = (outs Variadic<Fabric_AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_FuOp
    : Fabric_Op<
          "fu", [IsolatedFromAbove, SingleBlockImplicitTerminator<"YieldOp">,
                 DeclareOpInterfaceMethods<RegionKindInterface>,
                 DeclareOpInterfaceMethods<Symbol, ["isOptionalSymbol"]>]> {
  let summary = "fabric function unit (FU): a graph-region tile of fabric.op / "
                "mux / demux";
  let description = [{
    Container for one CGRA-style functional unit. The op exists in two
    disjoint syntactic forms by `sym_name` presence:

    Anonymous form (definition + use combined). Carries variadic SSA
    operands (with optional per-operand `to <inner-type>` clause for the
    FU input boundary's high-bit truncation) and variadic SSA results.
    The body's entry block argument types are the inner port types and
    the `fabric.yield` terminator supplies the values that match the
    FU's outer result types.

      %r = fabric.fu(%fa = %a : !fabric.bits<W> [to !fabric.bits<F>])
                 -> !fabric.bits<W> { ... fabric.yield %v : !fabric.bits<W> }

    Named template form (declaration only). Has zero SSA operands and
    zero SSA results; the port signature is captured in the
    `function_type` attribute. The body's entry block carries the input
    port types as block arguments and the `fabric.yield` terminator
    supplies values matching the function-type results. Actual usage of
    a named FU goes through `fabric.instantiate`.

      fabric.fu @F (!fabric.bits<W>, !fabric.bits<W>) -> !fabric.bits<W> {
      ^bb0(%fa: !fabric.bits<W>, %fb: !fabric.bits<W>):
        %v = fabric.op [@arith.muli] (%fa, %fb)
             : (!fabric.bits<W>, !fabric.bits<W>) -> !fabric.bits<W>
        fabric.yield %v : !fabric.bits<W>
      }

    Constraints:
      * The body must contain at least one `fabric.op`.
      * `fabric.fu` cannot be nested.
      * `fabric.fifo` is not permitted in the body.
      * The FU itself carries no `[hw_params]` / `{sw_configs}`;
        all configuration lives on the inner ops.
      * For each input port `i` (anonymous form only), the outer operand
        width must be greater than or equal to the inner block-arg width
        (`outer >= inner`). The named form has no outer operands.
      * Symmetric output-side boundary widening at the `fabric.yield`
        terminator (anonymous form): `fabric.yield %v : !fabric.bits<inner>
        to !fabric.bits<outer>` declares low-bit-aligned widening when
        `inner <= outer`. The high `(outer - inner)` bits are zero-filled
        at the FU boundary on each token; the outer type matches the
        FU's declared outer result type. Without the `to` clause, inner
        equals outer (strict).
  }];

  let arguments = (ins OptionalAttr<SymbolNameAttr>:$sym_name,
      OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
      OptionalAttr<Fabric_FuCapabilityDomainAttr>:$capability_templates,
      Variadic<Fabric_BitsType>:$inputs);
  let results = (outs Variadic<Fabric_BitsType>:$outputs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_InstantiateOp
    : Fabric_Op<
          "instantiate", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary =
      "fabric instantiation: bind a previously-defined fabric resource "
      "symbol into the current scope as a fresh hardware instance";
  let description = [{
    `fabric.instantiate` references a previously-defined fabric symbol
    (`fabric.module`, `fabric.pe`, `fabric.switch`, `fabric.mem`, or
    `fabric.fu`) and creates a fresh hardware instance with its own SSA
    inputs and results. The symbol target supplies the hardware shape
    (port count and per-port type); the instantiation site supplies the
    SSA inputs and consumer-side result SSA values.

    Allowed target kinds depend on the parent of the `fabric.instantiate`:
      * builtin top-level module: none.
      * `fabric.module` body: `fabric.module` (sibling top-level), or a
        named `fabric.pe`, `fabric.switch`, or `fabric.mem` defined inside
        the same module body.
      * `fabric.pe` body: a named `fabric.fu` defined inside the same
        pe body.

    Width relaxation at the input side mirrors the existing connection-
    point rule: the SSA operand type may differ in width from the target's
    declared input port type, expressed via the `to <inner-type>` clause.
    Differing destination types are retained in the ODS-owned typed
    `inner_input_types` property.
    The `domain_slot_bindings` property is the authoring-only canonical
    child-to-parent Clock/Reset slot relation for a Module target. It is
    required and non-empty for Module targets; non-Module targets use the
    empty relation.
    `memref` operands must match exactly. Output direction is strict:
    each result type must equal the target's declared output port type.

    Forward and self-references are forbidden.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
      Variadic<Fabric_ModulePortType>:$inputs,
      Fabric_InnerInputTypesProp:$inner_input_types,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$domain_slot_bindings);
  let results = (outs Variadic<Fabric_ModulePortType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_BoundaryOp : Fabric_Op<"boundary", [Pure]> {
  let summary =
      "fabric boundary op: bridge between the spatial bits domain and the "
      "temporal bits_tag domain (direction = s2t | t2t | t2s)";
  let description = [{
    Single boundary op carrying a mandatory `direction` predicate that
    selects one of three transition shapes between the spatial
    `!fabric.bits<BW>` domain and the temporal `!fabric.bits_tag<BW, TW>`
    domain. The bracketed `[<direction>]` predicate is printed/parsed in
    the same style as `fabric.pe [spatial|temporal]`.

    See `docs/spec-fabric-boundary.md` for the full per-direction
    surface syntax (s2t general / s2t constant-tag / t2t LUT / t2s split
    / t2s drop-tag).

    Allowed only inside a `fabric.module` body. The direction-specific
    relationships between destination input-port and result widths remain
    exact. An incoming anonymous connection may use `source-type to
    destination-port-type` for same-kind `bits` or `bits_tag` width
    normalization; differing destination types are retained in the ODS-owned
    typed `inner_input_types` property. See
    `docs/spec-fabric-boundary.md` for the per-direction verifier rules and
    the t2t LUT config-mem encoding.
  }];

  let arguments = (ins Variadic<Fabric_AnyType>:$inputs,
      Fabric_BoundaryDirectionAttr:$direction,
      Fabric_InnerInputTypesProp:$inner_input_types,
      OptionalAttr<ArrayAttr>:$hw_params,
      OptionalAttr<DictionaryAttr>:$sw_configs);
  let results = (outs Variadic<Fabric_AnyType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_SwitchOp
    : Fabric_Op<
          "switch", [DeclareOpInterfaceMethods<Symbol, ["isOptionalSymbol"]>,
                     Pure]> {
  let summary =
      "fabric switch: configurable routing crossbar with a schedule predicate "
      "(spatial or temporal)";
  let description = [{
    Leaf-level routing op with variadic SSA inputs and variadic SSA results.
    Carries a mandatory `schedule` predicate (`spatial` | `temporal`) and a
    `connectivity_table` describing the static physical fabric. The op
    exists in two disjoint syntactic forms by `sym_name` presence:

    Anonymous form has variadic SSA operands and variadic SSA results. Its
    input type list accepts `source-type to destination-port-type` pairs for
    same-kind `bits` or `bits_tag` width normalization; differing destination
    types are retained in the ODS-owned typed `inner_input_types` property.
    Named template form has zero SSA operands and zero SSA results in the
    host scope; the port signature is captured in the `function_type`
    attribute. Hardware parameters are carried in `hw_params` (an
    ArrayAttr wrapping a single DictionaryAttr) printed in `[ ... ]`
    after the SSA operand list. Software configuration is carried in
    `sw_configs` (a DictionaryAttr) printed in `{ ... }` after the
    hardware parameters.

    See `docs/spec-fabric-switch.md` for the full per-schedule rules
    (connectivity_table format, MSB-on-left bit-string convention,
    route_table layout, tag-driven temporal trigger semantics, etc.).
  }];

  let arguments = (ins Variadic<Fabric_AnyType>:$inputs,
      OptionalAttr<SymbolNameAttr>:$sym_name,
      OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
      Fabric_ScheduleAttr:$schedule,
      Fabric_InnerInputTypesProp:$inner_input_types,
      OptionalAttr<ArrayAttr>:$hw_params,
      OptionalAttr<DictionaryAttr>:$sw_configs);
  let results = (outs Variadic<Fabric_AnyType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_MemOp
    : Fabric_Op<
          "mem", [DeclareOpInterfaceMethods<Symbol, ["isOptionalSymbol"]>]> {
  let summary = "fabric memory capability occurrence";
  let description = [{
    One typed memory_contract owns the optional Operation Engine, optional
    Local Memory Service, and endpoint positions. The function_type remains
    the sole owner of endpoint kinds and widths. An Operation Engine owns its
    exact Spatial or Temporal schedule; a storage-only occurrence has none.

    Anonymous operands contain all manager/requester memrefs, followed by
    per-load `(addr, ctrl)` groups and per-store `(addr, data, ctrl)` groups.
    Results contain all subordinate/provider memrefs, followed by per-load
    `(data, done)` groups and per-store `done` results. Endpoint counts are
    derived from the signature and are not repeated as attributes. Incoming
    transport operands may use `source-type to destination-port-type` for
    same-kind width normalization.

    Named template form has zero SSA operands and zero SSA results in
    the host scope; the port signature is captured in `function_type`.
    `memory_operation_ports` contains the strict canonical byte encoding of
    every physical operation port. `hw_params` remains legacy authoring
    shorthand and cannot coexist with that exact inventory. It owns no
    occurrence, schedule, endpoint classification, or Local Memory Service
    fact. The registered authoring schema is closed. Canonical finalization may
    add only the typed entity/template IDs and the exact embedded ResourceContract
    record; all other discardable attributes are rejected.

    See `docs/spec-fabric-mem.md` for the full per-schedule rules.
  }];

  let arguments = (ins Variadic<Fabric_ModulePortType>:$inputs,
      OptionalAttr<SymbolNameAttr>:$sym_name,
      OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
      Fabric_MemoryContractAttr:$memory_contract,
      Fabric_InnerInputTypesProp:$inner_input_types,
      OptionalAttr<ArrayAttr>:$hw_params,
      OptionalAttr<ArrayAttr>:$memory_operation_ports);
  let results = (outs Variadic<Fabric_ModulePortType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_SystemOp : Fabric_Op<"system", [IsolatedFromAbove, SingleBlock,
                                           NoTerminator, Symbol]> {
  let summary = "typed system-level heterogeneous hardware root";
  let description = [{
    The single declarative architecture root. Its body has no block arguments,
    SSA values, CFG successors, symbol table, or runtime terminator. Only the
    closed `fabric.system.*` child operation catalog is admitted.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def Fabric_SystemHostCoreOp
    : Fabric_Op<"system.host_core", [HasParent<"::fabric::SystemOp">]> {
  let summary = "InstructionCore-only system compute entity";
  let description = [{
    One HostCore with exact InstructionCore architecture and
    microarchitecture records. `entity_id` is absent during authoring and is
    assigned by Fabric canonical finalization.
  }];

  let arguments = (ins OptionalAttr<Fabric_EntityIdAttr>:$entity_id,
      DenseI8ArrayAttr:$architecture, DenseI8ArrayAttr:$microarchitecture);

  let assemblyFormat = [{
    `architecture` `=` $architecture
    `microarchitecture` `=` $microarchitecture attr-dict
  }];
  let hasVerifier = 1;
}

def Fabric_SystemAccCoreOp
    : Fabric_Op<"system.acc_core", [HasParent<"::fabric::SystemOp">]> {
  let summary = "coupled InstructionCore and SpatialCore system entity";
  let description = [{
    One AccCore with exact InstructionCore architecture and microarchitecture
    records plus one compact ImportedModule target for its SpatialCore.
  }];

  let arguments = (ins OptionalAttr<Fabric_EntityIdAttr>:$entity_id,
      DenseI8ArrayAttr:$architecture, DenseI8ArrayAttr:$microarchitecture,
      DenseI8ArrayAttr:$spatial_core);

  let assemblyFormat = [{
    `architecture` `=` $architecture
    `microarchitecture` `=` $microarchitecture
    `spatial_core` `=` $spatial_core attr-dict
  }];
  let hasVerifier = 1;
}

def Fabric_SystemMemoryServiceOp
    : Fabric_Op<"system.memory_service", [HasParent<"::fabric::SystemOp">]> {
  let summary = "system-addressed memory service entity";
  let description = [{
    One exact MemoryServiceContractRecord interpreted in the System owner
    context. The record remains the sole owner of regions, service admission,
    resource state, timing, progress, and grant semantics.
  }];

  let arguments = (ins OptionalAttr<Fabric_EntityIdAttr>:$entity_id,
      Fabric_MemoryServiceContractAttr:$service_contract);

  let assemblyFormat = [{
    `contract` `=` $service_contract attr-dict
  }];
  let hasVerifier = 1;
}

def Fabric_SystemServiceEndpointOp
    : Fabric_Op<"system.service_endpoint", [HasParent<"::fabric::SystemOp">]> {
  let summary = "typed Canonical Service endpoint entity";
  let description = [{
    One endpoint owned by an exact System entity. Its capability set binds the
    closed Canonical Service kinds to typed accepted domains and one exact
    rate, outstanding-capacity, and progress contract.
  }];

  let arguments = (ins OptionalAttr<Fabric_EntityIdAttr>:$entity_id,
      DenseI8ArrayAttr:$owner, DenseI8ArrayAttr:$capabilities,
      OptionalAttr<TypeAttr>:$carrier_type);

  let assemblyFormat = [{
    `owner` `=` $owner `capabilities` `=` $capabilities
    (`carrier` `=` $carrier_type^)? attr-dict
  }];
  let hasVerifier = 1;
}

def Fabric_SystemServiceLegCarrierAttachmentOp
    : Fabric_Op<"system.service_leg_carrier_attachment",
                [HasParent<"::fabric::SystemOp">]> {
  let summary = "memory-service leg to candidate transport carriers";
  let description = [{
    One Fabric-owned structural relation from an exact memory-service endpoint,
    Canonical Service kind, and schema-local leg ordinal to a non-empty
    canonical carrier set. The encoded record has no EntityId.
  }];

  let arguments = (ins DenseI8ArrayAttr:$record);
  let assemblyFormat = "`record` `=` $record attr-dict";
  let hasVerifier = 1;
}

def Fabric_SystemServiceTransformOp
    : Fabric_Op<"system.service_transform", [HasParent<"::fabric::SystemOp">]> {
  let summary = "closed typed System memory-service transform";
  let description = [{
    One complete input/output relation and one AddressOffset, AddressMaskXor,
    StaticInterleave, or CoherentMemory contract. Identity remains a direct
    connection and has no transform operation.
  }];

  let arguments = (ins OptionalAttr<Fabric_EntityIdAttr>:$entity_id,
      DenseI8ArrayAttr:$contract);

  let assemblyFormat = "`contract` `=` $contract attr-dict";
  let hasVerifier = 1;
}

def Fabric_SystemExternalBoundaryOp
    : Fabric_Op<"system.external_boundary", [HasParent<"::fabric::SystemOp">]> {
  let summary = "external System interface owner";
  let description = [{
    One external interface grouping. Its complete physical interface is
    derived from the SystemServiceEndpoint entities that name this entity as
    owner; the boundary stores no parallel endpoint inventory.
  }];

  let arguments = (ins OptionalAttr<Fabric_EntityIdAttr>:$entity_id);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Fabric_SystemHardwareDomainOp
    : Fabric_Op<"system.hardware_domain", [HasParent<"::fabric::SystemOp">]> {
  let summary = "closed typed System hardware domain";
  let description = [{
    One Clock, Reset, Power, Address, or MemoryConsistency domain. The closed
    contract variant owns the kind and the canonical member sequence retains
    exact Fabric inventory-owner references.
  }];

  let arguments = (ins OptionalAttr<Fabric_EntityIdAttr>:$entity_id,
      DenseI8ArrayAttr:$contract);

  let assemblyFormat = "`contract` `=` $contract attr-dict";
  let hasVerifier = 1;
}

def Fabric_SystemTransportResourceOp
    : Fabric_Op<
          "system.transport_resource", [HasParent<"::fabric::SystemOp">]> {
  let summary = "typed System transport capacity and state owner";
  let description = [{
    One independently meaningful transport resource with exact token port
    types and one complete ResourceContractRecord. An optional crossing record
    is the only legal clock-domain crossing carried by this resource.
  }];

  let arguments = (ins OptionalAttr<Fabric_EntityIdAttr>:$entity_id,
      TypeAttrOf<FunctionType>:$function_type,
      DenseI8ArrayAttr:$resource_contract,
      OptionalAttr<DenseI8ArrayAttr>:$clock_crossing,
      OptionalAttr<UnitAttr>:$configuration_selected_patterns);

  let assemblyFormat = [{
    `ports` `=` $function_type
    `contract` `=` $resource_contract
    (`crossing` `=` $clock_crossing^)? attr-dict
  }];
  let hasVerifier = 1;
}

def Fabric_SystemTransferPatternOp
    : Fabric_Op<"system.transfer_pattern", [HasParent<"::fabric::SystemOp">]> {
  let summary = "one atomic System transport traversal pattern";
  let description = [{
    One ingress, a canonical non-empty egress set, and the exact owner-local
    UsePattern whose ResourceContract owns atomic claims, eligibility, timing,
    progress, and grant behavior.
  }];

  let arguments = (ins DenseI8ArrayAttr:$contract);
  let assemblyFormat = "`contract` `=` $contract attr-dict";
  let hasVerifier = 1;
}

def Fabric_SystemConnectionOp
    : Fabric_Op<"system.connection", [HasParent<"::fabric::SystemOp">]> {
  let summary = "fixed directed System point connection";
  let description = [{
    One exact point-to-point Transport or MemoryService connection. Transport
    connects an output to an input. MemoryService connects a manager to a
    subordinate. It has no identity, capacity, buffering, arbitration, or
    hidden crossing behavior; independently meaningful behavior is an
    explicit resource or service transform.
  }];

  let arguments = (ins DenseI8ArrayAttr:$source,
                       DenseI8ArrayAttr:$destination,
                       OptionalAttr<UnitAttr>:$memory_service);

  let assemblyFormat = [{
    `source` `=` $source `destination` `=` $destination attr-dict
  }];
  let hasVerifier = 1;
}

def Fabric_SystemSpatialAttachmentOp
    : Fabric_Op<
          "system.spatial_attachment", [HasParent<"::fabric::SystemOp">]> {
  let summary = "module boundary to SpatialCore occurrence attachment";
  let description = [{
    One structural, one-to-one correspondence between an ImportedModule
    boundary endpoint and the matching endpoint of one AccCore's SpatialCore
    occurrence. Direction, plane, type, and service role remain derived from
    the referenced endpoints.
  }];

  let arguments = (ins DenseI8ArrayAttr:$module_endpoint,
      DenseI8ArrayAttr:$spatial_endpoint,
      OptionalAttr<DenseI8ArrayAttr>:$service_endpoint);

  let assemblyFormat = [{
    `module_endpoint` `=` $module_endpoint
    `spatial_endpoint` `=` $spatial_endpoint
    (`service_endpoint` `=` $service_endpoint^)? attr-dict
  }];
  let hasVerifier = 1;
}

def Fabric_InterconnectImplementationOp
    : Fabric_Op<"interconnect_implementation",
                [IsolatedFromAbove, SingleBlock, NoTerminator, Symbol]> {
  let summary = "typed concrete refinement of one Fabric System interconnect";
  let description = [{
    One content-addressed InterconnectImplementation root. The closed protocol
    schema selects the only legal operation catalog in `implementation`; the
    `refinements` region contains the complete architecture-to-protocol
    relation. The exact refined System remains an artifact-envelope dependency
    and is deliberately not duplicated as an operation attribute.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
      Fabric_InterconnectProtocolSchemaAttr:$protocol_schema);
  let regions = (region SizedRegion<1>:$implementation,
                         SizedRegion<1>:$refinements);

  let assemblyFormat = [{
    $sym_name `schema` `=` $protocol_schema attr-dict-with-keyword
    `implementation` $implementation `refinements` $refinements
  }];
  let hasVerifier = 1;
}

def Fabric_InterconnectGem5EventEndpointOp
    : Fabric_Op<"interconnect.gem5_event.endpoint",
                [HasParent<"::fabric::InterconnectImplementationOp">]> {
  let summary = "one typed gem5 event-transport protocol endpoint";
  let arguments = (ins I64Attr:$ordinal,
      Fabric_InterconnectEndpointDirectionAttr:$direction);
  let assemblyFormat = "$ordinal $direction attr-dict";
  let hasVerifier = 1;
}

def Fabric_InterconnectGem5EventResourceOp
    : Fabric_Op<"interconnect.gem5_event.resource",
                [HasParent<"::fabric::InterconnectImplementationOp">]> {
  let summary = "one typed gem5 event-transport resource state";
  let arguments = (ins I64Attr:$ordinal);
  let assemblyFormat = "$ordinal attr-dict";
  let hasVerifier = 1;
}

def Fabric_InterconnectGem5EventTransferOp
    : Fabric_Op<"interconnect.gem5_event.transfer",
                [HasParent<"::fabric::InterconnectImplementationOp">]> {
  let summary = "one typed gem5 event-transport transfer";
  let arguments = (ins I64Attr:$ordinal, I64Attr:$ingress,
      DenseI64ArrayAttr:$egresses, DenseI64ArrayAttr:$resources);
  let assemblyFormat = [{
    $ordinal `ingress` `=` $ingress `egresses` `=` $egresses
    `resources` `=` $resources attr-dict
  }];
  let hasVerifier = 1;
}

def Fabric_InterconnectGem5EventConfigurationFieldOp
    : Fabric_Op<"interconnect.gem5_event.configuration_field",
                [HasParent<"::fabric::InterconnectImplementationOp">]> {
  let summary = "one typed gem5 event-transport configuration field";
  let arguments = (ins I64Attr:$ordinal);
  let assemblyFormat = "$ordinal attr-dict";
  let hasVerifier = 1;
}

def Fabric_InterconnectRefinementOp
    : Fabric_Op<"interconnect.refinement",
                [HasParent<"::fabric::InterconnectImplementationOp">]> {
  let summary = "one closed architecture-to-protocol refinement record";
  let arguments = (ins Fabric_InterconnectRefinementKindAttr:$kind,
      DenseI8ArrayAttr:$architecture_ref,
      DenseI64ArrayAttr:$protocol_refs);
  let assemblyFormat = [{
    $kind `architecture` `=` $architecture_ref
    `protocol` `=` $protocol_refs attr-dict
  }];
  let hasVerifier = 1;
}

def Fabric_YieldOp : Fabric_Op<"yield", [Terminator, Pure]> {
  let summary =
      "terminator of a fabric.fu, named fabric.pe, or fabric.module region";
  let description = [{
    Closes the enclosing `fabric.fu`, named `fabric.pe`, or `fabric.module`
    body.

    Inside `fabric.fu`, `$values` supplies the values that leave the FU
    on its results; the number and types must match the parent op's
    result count and types exactly.

    Inside a named `fabric.pe`, the terminator carries no values. The PE's
    logical port signature is owned by `function_type` alone and the
    configured output selectors choose each result's internal source, so
    the terminator never restates the result ports. An anonymous
    `fabric.pe` has no terminator at all.

    Inside `fabric.module`, `$values` supplies the module's declared
    output values. The number and per-value types must match the
    module's declared result list under the width-relaxation rule
    documented in `docs/spec-fabric-module.md` (low-bit alignment, same
    fabric kind; memref must match exactly). The optional per-value
    `to <module-result-type>` clause makes the relaxation explicit:

      fabric.yield %v : !fabric.bits<32> to !fabric.bits<16>

  }];

  // Yield operands are the SSA values being yielded. They must be one of
  // the allowed module port types when this yield closes a fabric.module
  // body, and !fabric.bits<W> when it closes a fabric.fu body. The op
  // accepts the union here; `verify` enforces the per-parent rule.
  let arguments = (ins Variadic<Fabric_ModulePortType>:$values,
      OptionalAttr<ArrayAttr>:$declared_types);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [OpBuilder<(ins CArg<"::mlir::ValueRange", "{}">:$values), [{
      build($_builder, $_state, values, ::mlir::ArrayAttr());
    }]>];
}

#endif // FABRIC_OPS_TD
